"""Template file showing unit tests for MyManifold.

MyManifold is the template manifold defined in:
geomstats/geometry/_my_manifold.py.

For additional guidelines on how to contribute to geomstats, visit:
https://geomstats.github.io/contributing.html#contributing-code-workflow

To run these tests:
- Install packages from geomstats/dev-requirements.txt
- In command line, run:
```nose2 tests.test__my_manifold``` to run all the tests of this file
- In command line, run:
```nose2 tests.test__my_manifold.TestMyManifold.test_dimension```
to run the test `test_dimension` only.

To run these tests using different backends (numpy, pytorch or tensorflow):
- Install packages from geomstats/opt-requirements.tct
In command line, select the backend of interest with:
```export GEOMSTATS_BACKEND=numpy```
 or ```export GEOMSTATS_BACKEND=pytorch```
 or ```export GEOMSTATS_BACKEND=tensorflow```
 and repeat the steps from the previous paragraph.
"""

# Import the tests module
import geomstats.backend as gs
import geomstats.tests
# Import the manifold to be tested
from geomstats.geometry._my_manifold import MyManifold


class TestMyManifold(geomstats.tests.TestCase):
    """Class testing the methods of MyManifold.

    In the class TestMyManifold, each test method:
    - needs to start with `test_`
    - represents a unit-test, i.e. tests one and only one method
    or attribute of the class MyManifold,
    - ends with the line: `self.assertallclose(result, expected)`, see below.
    """
    def setUp(self):
        """setUp method.

        Use the setUp method to define variables that stay constant
        during all tests. For example, here we test the
        4-dimensional manifold of the class MyManifold.
        """
        self.dimension = 4
        self.another_parameter = 3
        self.manifold = MyManifold(
            dim=self.dimension, another_parameter=3)

    def test_dimension(self):
        """Test dimension.

        The method test_dimension tests the `dim` attribute.
        """

        result = self.manifold.dim
        expected = self.dimension
        # Each test ends with the following syntax, comparing
        # the result with the expected result, using self.assertAllClose
        self.assertallclose(result, expected)

    def test_belongs(self):
        """Test belongs.

        The method test_belongs tests the `belongs` method.

        Note that arrays are defined using geomstats backend
        through the prefix `gs.`.
        This allows the code to be tested simultaneously in numpy,
        pytorch and tensorflow. `gs.` is the equivalent of numpy's `np.` and
        most of numpy's functions are available with `gs.`.
        """
        point = gs.array([1., 2., 3.])
        result = self.manifold.belongs(point)
        expected = False
        self.assertAllClose(result, expected)
