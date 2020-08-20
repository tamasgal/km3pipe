# Filename: test_shell.py
# pylint: disable=locally-disabled,C0111,R0904,C0103

from km3pipe.testing import TestCase
from km3pipe.shell import Script

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2017, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestScript(TestCase):
    def test_add(self):
        s = Script()
        s.add("a")
        s.add("b")

    def test_str(self):
        s = Script()
        s.add("a")
        s.add("b")
        self.assertEqual("a\nb", str(s))

    def test_clear(self):
        s = Script()
        s.add("a")
        s.clear()
        self.assertEqual("", str(s))

    def test_add_two_argument_command(self):
        s = Script()
        s._add_two_argument_command("command", "a", "b")
        self.assertEqual("command a b", str(s))

    def test_add_two_argument_command_multiple_times(self):
        s = Script()
        s._add_two_argument_command("cmd1", "a", "b")
        s._add_two_argument_command("cmd2", "c", "d")
        s._add_two_argument_command("cmd3", "e", "f")
        self.assertEqual("cmd1 a b\ncmd2 c d\ncmd3 e f", str(s))

    def test_cp(self):
        s = Script()
        s.cp("a", "b")
        self.assertEqual("cp a b", str(s))

    def test_mv(self):
        s = Script()
        s.mv("a", "b")
        self.assertEqual("mv a b", str(s))

    def test_echo(self):
        s = Script()
        s.echo("test")
        self.assertEqual('echo "test"', str(s))

    def test_separator(self):
        s = Script()
        s.separator()
        self.assertEqual('echo "' + "=" * 42 + '"', str(s))

    def test_mkdir(self):
        s = Script()
        s.mkdir("/path/to/file")
        self.assertEqual('mkdir -p "/path/to/file"', str(s))

    def test_iget(self):
        s = Script()
        s.iget("/path/to/file")
        self.assertEqual('iget -v "/path/to/file"', str(s))

    def test_combining_scripts(self):
        s1 = Script()
        s2 = Script()
        s1.add("a")
        s1.add("b")
        s2.add("c")
        s2.add("d")
        assert "a\nb\nc\nd" == str(s1 + s2)
