#!/usr/bin/env zsh

export PAN="--from markdown+link_attributes+multiline_tables+simple_tables+definition_lists+backtick_code_blocks+fenced_code_attributes+implicit_figures"
export PAN=

pandoc $PAN -f markdown -t rst $@
