#!/bin/bash
set -e

echo "Installing Python dependencies..."

# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install requirements with explicit wheel preference
pip install --only-binary :all: -r requirements.txt || pip install -r requirements.txt

echo "Build completed successfully!"
