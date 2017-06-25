from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .dataset import Dataset

class KTHData(Dataset):
  """KTH dataset."""

  def __init__(self, subset):
    super(KTHData, self).__init__('KTH', subset)

  def num_classes(self):
    """Returns the number of classes in the data set."""
    return 6

  def num_examples_per_epoch(self):
    """Returns the number of examples in the data subset."""
    if self.subset == 'train':
      return 2981
    if self.subset == 'validation':
      return 50

  def download_message(self):
    """Instruction to download and extract the tarball from KTH website."""

    print('Failed to find any KTH %s files'% self.subset)
    print('')
    print('If you have already downloaded and processed the data, then make '
          'sure to set --data_path to point to the directory containing the '
          'location of the sharded TFRecords.\n')
