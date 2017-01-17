# coding=utf-8
# Filename: test_controlhost.py
# pylint: disable=locally-disabled,C0111,R0904,R0201,C0103,W0612
"""
Unit tests for the controlhost module.

"""
from __future__ import absolute_import, print_function

from km3pipe.controlhost import Tag, Message, Prefix

from km3pipe.testing import TestCase, MagicMock

from km3pipe.db import DBManager, DOMContainer
from km3pipe.logger import logging

__author__ = "Tamas Gal"
__copyright__ = "Copyright 2016, Tamas Gal and the KM3NeT collaboration."
__credits__ = []
__license__ = "MIT"
__maintainer__ = "Tamas Gal"
__email__ = "tgal@km3net.de"
__status__ = "Development"

class TestTag(TestCase):
    def test_init(self):
        tag = Tag()

    def test_empty_tag_has_correct_length(self):
        tag = Tag()
        self.assertEqual(Tag.SIZE, len(tag))

    def test_tag_has_correct_length(self):
        for tag_name in (b'foo', b'bar', b'baz', b'1'):
            tag = Tag(tag_name)
            self.assertEqual(Tag.SIZE, len(tag))

    def test_tag_with_invalid_length_raises_valueerror(self):
        self.assertRaises(ValueError, Tag, '123456789')

    def test_tag_has_correct_data(self):
        tag = Tag(b'foo')
        self.assertEqual(b'foo\x00\x00\x00\x00\x00', tag.data)
        tag = Tag('abcdefgh')
        self.assertEqual('abcdefgh', tag.data)

    def test_tag_has_correct_string_representation(self):
        tag = Tag(b'foo')
        self.assertEqual('foo', str(tag))



class TestPrefix(TestCase):
    def test_init(self):
        prefix = Prefix(b'foo', 1)


class TestMessage(TestCase):
    def test_init(self):
        message = Message('')
