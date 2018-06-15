# Filename: test_testing.py
# pylint: disable=locally-disabled,C0111,R0904,C0103
from km3pipe.testing import TestCase, surrogate

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


def imports():
    import my
    import my.module
    import my.module.one
    import my.module.two
    from my import module
    from my.module import one, two
    return True


class TestSurrogateModuleStubs(TestCase):
    def test_surrogating(self):
        @surrogate('my')
        @surrogate('my.module.one')
        @surrogate('my.module.two')
        def stubbed():
            imports()

        try:
            stubbed()
        except Exception as e:
            raise Exception('Modules are not stubbed correctly: %r' % e)

        with self.assertRaises(ImportError):
            imports()

    def test_context_manager(self):
        with surrogate('my'):
            with surrogate('my.module.one'):
                with surrogate('my.module.two'):
                    imports()
