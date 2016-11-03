#!/bin/bash
TARGET_PYTHON_VERSION="2.7.12"
TEMP_SOURCE_FILE=".tmp_km3pipe"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${RED}Installing pyenv.${NC}"
git clone https://github.com/yyuu/pyenv.git ~/.pyenv
echo -e "${GREEN}Done.${NC}"

echo -e "${RED}Installing pyenv-virtualenv${NC}"
git clone https://github.com/yyuu/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv
echo -e "${GREEN}Done.${NC}"

read -r -d '' SHELL_RC << EOM
# pyenv
export PYENV_ROOT="\${HOME}/.pyenv"
export PATH="\${PYENV_ROOT}/bin:\$PATH"
eval "\$(pyenv init -)"
eval "\$(pyenv virtualenv-init -)"
EOM

echo -e "${RED}Appending pyenv init commands to your shell rc.${NC}"
echo "${SHELL_RC}" >> ${HOME}/.zshrc_local
echo "${SHELL_RC}" >> bashrc
echo -e "${GREEN}Done.${NC}"

echo -e "${RED}Setting environment variables for current shell session.${NC}"
echo "${SHELL_RC}" > ${TEMP_SOURCE_FILE}
source ${TEMP_SOURCE_FILE}
rm ${TEMP_SOURCE_FILE}
echo -e "${GREEN}Done.${NC}"

echo -e "${RED}Installing Python ${TARGET_PYTHON_VERSION}.${NC}"
pyenv install ${TARGET_PYTHON_VERSION}
echo -e "${GREEN}Done.${NC}"

echo -e "${RED}Upgrading pip and setuptools.${NC}"
pip install -U pip setuptools
echo -e "${GREEN}Done.${NC}"

echo -e "${RED}Installing numpy and cython.${NC}"
pip install -U numpy cython
echo -e "${GREEN}Done.${NC}"

echo -e "${RED}Installing KM3Pipe.${NC}"
pip install km3pipe
echo -e "${GREEN}Done.${NC}"

echo -e "${GREEN}Done! Starting shiny new shell session.${NC}"
exec $SHELL
