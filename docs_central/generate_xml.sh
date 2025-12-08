#!/bin/bash
SRC_ROOT="../src/mobile_manipulation_packages"
PACKAGES=("manipulation" "mobile_bringup" "mobile_manipulation_interfaces" "navigation" "storage_manager" "task_planning")

echo "=== Gerando XML para todos os pacotes ==="

for PKG in "${PACKAGES[@]}"; do
    mkdir -p xml_output/$PKG
    (
        cat <<EOF
PROJECT_NAME = "$PKG"
INPUT = $SRC_ROOT/$PKG
RECURSIVE = YES
OUTPUT_DIRECTORY = xml_output/$PKG
XML_OUTPUT = xml
GENERATE_HTML = NO
GENERATE_LATEX = NO
GENERATE_XML = YES
EXTRACT_ALL = YES
EXTRACT_PRIVATE = YES
EOF
    ) | doxygen -
done