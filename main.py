# -*- coding: utf-8 -*-
"""Perform training and testing on VRD."""

from src.models import matranse_model


def main():
    """Train and test a network pipeline."""
    matranse_model.train_test()

if __name__ == "__main__":
    main()
