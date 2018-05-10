import glob
import gzip
import json
import random

IUPAC_CODES = list('ACDEFGHIKLMNPQRSTVWY*')


class SGISequenceGenerator(object):
    def __init__(
            self,
            filename_pattern,
            shard_count=None,
            shard_index=None,
            unbounded_iteration=True,
    ):
        self.filename_pattern = filename_pattern
        self.filenames = glob.glob(filename_pattern)
        if len(self.filenames) == 0:
            raise ValueError(
                "Could not find filenames according to pattern {}".format(filename_pattern)
            )

        if shard_count is not None:
            if shard_count > len(self.filenames):
                raise ValueError((
                    'shard_count must be <= the number of files for now. '
                    'requested {} shards but only found {} filenames'
                ).format(shard_count, len(self.filenames)))
            if shard_index is None:
                raise ValueError('if shard_count is not None, shard_index must not be None')

            self.filenames.sort()
            self.filenames = self.filenames[shard_index::shard_count]

        self.filename_index = 0
        self.data_handle = None

        self.reset_count = 0
        self.unbounded_iteration = unbounded_iteration

    def file_open(self, index):
        """
        Open the file located at index in self.filenames and sets handle to
        self.data_handle

        If file is gzip, open as decompressed stream
        """

        if self.data_handle:
            self.data_handle.close()

        filename = self.filenames[index]

        if filename.endswith('.gz'):
            self.data_handle = gzip.GzipFile(filename, 'rb')
        else:
            self.data_handle = open(filename, 'rb')

        # print("Opening " + filename)

    def reset(self):
        """
        Resets the starting index of this dataset to zero. Useful for calling
        repeated evaluations on the dataset without having to wrap around the
        last uneven minibatch. Not necessary when data is divisible by batch
        size
        """
        self.reset_count += 1
        self.filename_index = 0

        random.shuffle(self.filenames)
        self.file_open(self.filename_index)

    def next_file(self):
        """
        Point data_handle to next file (wrapping around at beginning if need
        be)
        """
        if self.filename_index == len(self.filenames) - 1:
            self.reset()
        else:
            self.filename_index += 1
            self.file_open(self.filename_index)
            # print 'opened file {} of {}'.format(
            #     self.filename_index, len(self.filenames)
            # )

    def read_next_line(self):
        """
        Read the next line from the currently open file.  If there are no more
        lines in that file, open the next one.
        """
        filename = self.filenames[self.filename_index]

        try:
            data = self.data_handle.readline()
        except IOError as e:
            print 'IO ERROR reading file: ' + filename
            data = []
        except:
            print 'Unknown ERROR reading file: ' + filename
            data = []

        if len(data) == 0:
            self.next_file()
            return (self.read_next_line())

        if isinstance(data, bytes):
            data = json.loads(data.decode('utf-8'))
        else:
            data = json.loads(data)

        return data

    def __iter__(self):
        """
        yield each record in the data set one time
        """
        self.reset()

        start_reset_count = self.reset_count
        while True:
            record = self.read_next_line()
            if not self.unbounded_iteration and start_reset_count != self.reset_count:
                break
            yield record
