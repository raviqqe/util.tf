import doctest
import unittest

from . import util



def main():
  doctest.testmod(util)
  unittest.main()


if __name__ == "__main__":
  main()
