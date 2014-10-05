try:
    import unittest2 as unittest
except ImportError:
    import unittest

try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

from km3pipe.decorators import remain_file_pointer


class TestRemainFilePointer(unittest.TestCase):

    def test_remains_file_pointer_in_function(self):
        dummy_file = StringIO('abcdefg')

        @remain_file_pointer
        def seek_into_file(file_obj):
            file_obj.seek(1, 0)

        dummy_file.seek(2, 0)
        self.assertEqual(2, dummy_file.tell())
        seek_into_file(dummy_file)
        self.assertEqual(2, dummy_file.tell())

    def test_remains_file_pointer_in_class_method(self):

        class FileSeekerClass(object):
            def __init__(self):
                self.dummy_file = StringIO('abcdefg')

            @remain_file_pointer
            def seek_into_file(self, file_obj):
                file_obj.seek(1, 0)

        fileseeker = FileSeekerClass()
        fileseeker.dummy_file.seek(2, 0)
        self.assertEqual(2, fileseeker.dummy_file.tell())
        fileseeker.seek_into_file(fileseeker.dummy_file)
        self.assertEqual(2, fileseeker.dummy_file.tell())
