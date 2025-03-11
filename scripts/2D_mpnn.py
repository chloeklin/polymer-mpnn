import sys
sys.path.append("..") 
import numpy as np
import pandas as pd
import tensorflow as tf
import nfp
from nfp.preprocessing.mol_preprocessor import SmilesPreprocessor, MolPreprocessor
from nfp.preprocessing.features import get_ring_size
from tensorflow.keras import layers

# Load the input data
# train_3D, valid_3D, test_3D_dft, test_3D_uff = pd.read_csv('../data/mol_train.csv'), pd.read_csv('../data/mol_valid.csv'), pd.read_csv('../data/mol_test.csv'), pd.read_csv('../data/mol_test_uff.csv')
train_2D, valid_2D, test_2D = pd.read_csv('../data/smiles_train.csv'), pd.read_csv('../data/smiles_valid.csv'), pd.read_csv('../data/smiles_test.csv')


# Define how to featurize the input molecules
def atom_featurizer(atom):
    """ Return an string representing the atom type
    """
    return str((
        atom.GetSymbol(),
        atom.GetIsAromatic(),
        get_ring_size(atom, max_size=6),
        atom.GetDegree(),
        atom.GetTotalNumHs(includeNeighbors=True)
    ))


def bond_featurizer(bond, flipped=False):
    """ Get a similar classification of the bond type.
    Flipped indicates which 'direction' the bond edge is pointing. """
    
    if not flipped:
        atoms = "{}-{}".format(
            *tuple((bond.GetBeginAtom().GetSymbol(),
                    bond.GetEndAtom().GetSymbol())))
    else:
        atoms = "{}-{}".format(
            *tuple((bond.GetEndAtom().GetSymbol(),
                    bond.GetBeginAtom().GetSymbol())))
    
    btype = str(bond.GetBondType())
    ring = 'R{}'.format(get_ring_size(bond, max_size=6)) if bond.IsInRing() else ''
    
    return " ".join([atoms, btype, ring]).strip()


smiles_preprocessor = SmilesPreprocessor(atom_features=atom_featurizer, bond_features=bond_featurizer,
                                  explicit_hs=False)
# mol_preprocessor = MolPreprocessor()

# Initially, the preprocessor has no data on atom types, so we have to loop over the 
# training set once to pre-allocate these mappings
for smiles in train_2D.smile.to_numpy():
    smiles_preprocessor(smiles, train=True)

# Construct the tf.data pipeline. There's a lot of specifying data types and
# expected shapes for tensorflow to pre-allocate the necessary arrays. But 
# essentially, this is responsible for calling the input constructor, batching 
# together multiple molecules, and padding the resulting molecules so that all
# molecules in the same batch have the same number of atoms (we pad with zeros,
# hence why the atom and bond types above start with 1 as the unknown class)

train_2D_dataset = tf.data.Dataset.from_generator(
    lambda: ((smiles_preprocessor(row.smile, train=False), 
              (row.gap, row.homo, row.lumo, row.spectral_overlap, row.homo_extrapolated,
               row.lumo_extrapolated, row.gap_extrapolated, row.optical_lumo_extrapolated))  # Multiple properties
             for i, row in train_2D.iterrows()),
    output_signature=(
        preprocessor.output_signature, 
        (tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32))  
)).cache().shuffle(buffer_size=200)\
  .padded_batch(batch_size=64)\
  .prefetch(tf.data.experimental.AUTOTUNE)


valid_2D_dataset = tf.data.Dataset.from_generator(
    lambda: ((smiles_preprocessor(row.smile, train=False), 
              (row.gap, row.homo, row.lumo, row.spectral_overlap, row.homo_extrapolated,
               row.lumo_extrapolated, row.gap_extrapolated, row.optical_lumo_extrapolated))  # Multiple properties
             for i, row in valid_2D.iterrows()),
    output_signature=(
        smiles_preprocessor.output_signature,
        (tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32))  
)).cache()\
    .padded_batch(batch_size=64)\
    .prefetch(tf.data.experimental.AUTOTUNE)

test_2D_dataset = tf.data.Dataset.from_generator(
    lambda: (smiles_preprocessor(smiles, train=False)
             for smiles in test_2D.smiles.to_numpy()),
    output_signature=(
        smiles_preprocessor.output_signature,
        (tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32),  
         tf.TensorSpec((), dtype=tf.float32))
  ))\
    .padded_batch(batch_size=64)\
    .prefetch(tf.data.experimental.AUTOTUNE)

## Define model
# Input layers
atom = layers.Input(shape=[None], dtype=tf.int64, name='atom')
bond = layers.Input(shape=[None], dtype=tf.int64, name='bond')
connectivity = layers.Input(shape=[None, 2], dtype=tf.int64, name='connectivity')

num_features = 128  # Controls the size of the model

# Convert from a single integer defining the atom state to a vector
# of weights associated with that class
atom_state = layers.Embedding(smiles_preprocessor.atom_classes, num_features,
                              name='atom_embedding', mask_zero=True)(atom)

# Ditto with the bond state
bond_state = layers.Embedding(smiles_preprocessor.bond_classes, num_features,
                              name='bond_embedding', mask_zero=True)(bond)

# Here we use our first nfp layer. This is an attention layer that looks at
# the atom and bond states and reduces them to a single, graph-level vector. 
# mum_heads * units has to be the same dimension as the atom / bond dimension
global_state = nfp.GlobalUpdate(units=128, num_heads=1)([atom_state, bond_state, connectivity])

for _ in range(3):  # Do the message passing
    new_bond_state = nfp.EdgeUpdate()([atom_state, bond_state, connectivity, global_state])
    bond_state = layers.Add()([bond_state, new_bond_state])
    
    new_atom_state = nfp.NodeUpdate()([atom_state, bond_state, connectivity, global_state])
    atom_state = layers.Add()([atom_state, new_atom_state])
    
    new_global_state = nfp.GlobalUpdate(units=128, num_heads=1)(
        [atom_state, bond_state, connectivity, global_state]) 
    global_state = layers.Add()([global_state, new_global_state])

    
# Since the final prediction has 8 property predictions, we 
# reduce the last global state to 8.
# predictions = layers.Dense(8)(global_state)
predictions = {
    "gap": layers.Dense(1, name="gap")(global_state),
    "homo": layers.Dense(1, name="homo")(global_state),
    "lumo": layers.Dense(1, name="lumo")(global_state),
    "spectral_overlap": layers.Dense(1, name="spectral_overlap")(global_state),
    "homo_extrapolated": layers.Dense(1, name="homo_extrapolated")(global_state),
    "lumo_extrapolated": layers.Dense(1, name="lumo_extrapolated")(global_state),
    "gap_extrapolated": layers.Dense(1, name="gap_extrapolated")(global_state),
    "optical_lumo_extrapolated": layers.Dense(1, name="optical_lumo_extrapolated")(global_state),
}


# Construct the tf.keras model
model = tf.keras.Model([atom, bond, connectivity], outputs=predictions)

# Initial learning rate
initial_learning_rate = 1E-3  # From the paper

# Define exponential decay schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, 
    decay_steps=1,  # Decay applied per epoch
    decay_rate=tf.math.exp(-2E-6),  # Matches paper's decay of 2 × 10^−6 per epoch
    staircase=False  # Smooth decay instead of discrete steps
)

# Adam optimizer with decaying learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


model.compile(loss='mae', optimizer=optimizer)

# Fit the model. The first epoch is slower, since it needs to cache
# the preprocessed molecule inputs
model.fit(train_2D_dataset, validation_data=valid_2D_dataset, epochs=500)