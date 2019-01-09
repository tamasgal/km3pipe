# Filename: test_math.py
# pylint: disable=locally-disabled,C0111,R0904,C0103

import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose)
import pytest

from km3pipe.testing import TestCase
from km3pipe.math import (
    angle_between, dist, pld3, com, zenith, azimuth, Polygon, IrregularPrism,
    rotation_matrix, SparseCone, space_angle, hsin, phi, theta, unit_vector,
    innerprod_1d, log_b, qeuler, qrot, qrot_yaw, intersect_3d
)

__author__ = ["Tamas Gal", "Moritz Lotze"]
__copyright__ = "Copyright 2016, KM3Pipe devs and the KM3NeT collaboration."
__credits__ = ["Thomas Heid"]
__license__ = "MIT"
__maintainer__ = ["Tamas Gal", "Moritz Lotze"]
__email__ = "tgal@km3net.de"
__status__ = "Development"


class TestMath(TestCase):
    def setUp(self):
        # self.vecs = np.array([[0., 1., 5.],
        #                       [1., 1., 4.],
        #                       [2., 1., 3.],
        #                       [3., 1., 2.],
        #                       [4., 1., 1.]])
        # self.v = (1, 2, 3)
        self.v = np.array([0.26726124, 0.53452248, 0.80178373])
        self.vecs = np.array([[0., 0.19611614, 0.98058068],
                              [0.23570226, 0.23570226, 0.94280904],
                              [0.53452248, 0.26726124, 0.80178373],
                              [0.80178373, 0.26726124, 0.53452248],
                              [0.94280904, 0.23570226, 0.23570226]])

    def test_phi(self):
        print(phi((1, 0, 0)))
        assert_almost_equal(0, phi((1, 0, 0)))
        assert_almost_equal(np.pi, phi((-1, 0, 0)))
        assert_almost_equal(np.pi / 2, phi((0, 1, 0)))
        assert_almost_equal(np.pi / 2 * 3, phi((0, -1, 0)))
        assert_almost_equal(np.pi / 2 * 3, phi((0, -1, 0)))
        assert_almost_equal(0, phi((0, 0, 0)))
        assert_almost_equal(phi(self.v), 1.10714872)
        assert_almost_equal(
            phi(self.vecs),
            np.array([
                1.57079633, 0.78539816, 0.46364761, 0.32175055, 0.24497866
            ])
        )

    def test_zenith(self):
        assert_allclose(np.pi, zenith((0, 0, 1)))
        assert_allclose(0, zenith((0, 0, -1)))
        assert_allclose(np.pi / 2, zenith((0, 1, 0)))
        assert_allclose(np.pi / 2, zenith((0, -1, 0)))
        assert_allclose(np.pi / 4 * 3, zenith((0, 1, 1)))
        assert_allclose(np.pi / 4 * 3, zenith((0, -1, 1)))
        assert_almost_equal(zenith(self.v), 2.5010703409103687)
        assert_allclose(
            zenith(self.vecs),
            np.array([
                2.94419709, 2.80175574, 2.50107034, 2.13473897, 1.80873745
            ])
        )

    def test_azimuth(self):
        self.assertTrue(np.allclose(np.pi, azimuth((1, 0, 0))))
        self.assertTrue(np.allclose(0, azimuth((-1, 0, 0))))

        print(azimuth((0, 1, 0)))
        print(azimuth((0, -1, 0)))
        print(azimuth((0, 0, 0)))
        print(azimuth(self.v))
        print(azimuth(self.vecs))
        self.assertTrue(np.allclose(np.pi / 2 * 3, azimuth((0, 1, 0))))
        self.assertTrue(np.allclose(np.pi / 2, azimuth((0, -1, 0))))
        self.assertTrue(np.allclose(np.pi, azimuth((0, 0, 0))))
        self.assertTrue(np.allclose(azimuth(self.v), 4.24874137138))
        self.assertTrue(
            np.allclose(
                azimuth(self.vecs),
                np.array([
                    4.71238898, 3.92699082, 3.60524026, 3.46334321, 3.38657132
                ])
            )
        )

    def test_theta(self):
        print(theta((0, 0, -1)))
        print(theta((0, 0, 1)))
        print(theta((0, 1, 0)))
        print(theta((0, -1, 0)))
        print(theta((0, 1, 1)))
        print(theta((0, -1, 1)))
        print(theta(self.v))
        print(theta(self.vecs))
        self.assertTrue(np.allclose(0, theta((0, 0, 1))))
        self.assertTrue(np.allclose(np.pi, theta((0, 0, -1))))
        self.assertTrue(np.allclose(np.pi / 2, theta((0, 1, 0))))
        self.assertTrue(np.allclose(np.pi / 2, theta((0, -1, 0))))
        self.assertTrue(np.allclose(0, theta((0, 1, 1))))
        self.assertTrue(np.allclose(0, theta((0, -1, 1))))
        self.assertTrue(np.allclose(theta(self.v), 0.64052231))
        self.assertTrue(
            np.allclose(
                theta(self.vecs),
                np.array([
                    0.19739554, 0.33983691, 0.64052231, 1.00685369, 1.3328552
                ])
            )
        )

    def test_unit_vector(self):
        v1 = (1, 0, 0)
        v2 = (1, 1, 0)
        v3 = (-1, 2, 0)
        assert np.allclose(v1, unit_vector(v1))
        assert np.allclose(np.array(v2) / np.sqrt(2), unit_vector(v2))
        assert np.allclose(np.array(v3) / np.sqrt(5), unit_vector(v3))

    def test_innerprod_1d(self):
        v1 = [[1, 2, 3], [4, 5, 6]]
        v2 = [[11, 22, 33], [44, 55, 66]]
        assert np.allclose([14, 77], innerprod_1d(v1, v2))

    def test_angle_between(self):
        v1 = (1, 0, 0)
        v2 = (0, 1, 0)
        v3 = (-1, 0, 0)
        self.assertAlmostEqual(0, angle_between(v1, v1))
        self.assertAlmostEqual(np.pi / 2, angle_between(v1, v2))
        self.assertAlmostEqual(np.pi, angle_between(v1, v3))
        self.assertAlmostEqual(angle_between(self.v, v1), 1.3002465638163236)
        self.assertAlmostEqual(angle_between(self.v, v2), 1.0068536854342678)
        self.assertAlmostEqual(angle_between(self.v, v3), 1.8413460897734695)
        self.assertTrue(
            np.allclose(
                angle_between(self.vecs, v1),
                np.array([
                    1.57079633, 1.3328552, 1.0068537, 0.64052231, 0.33983691
                ])
            )
        )
        self.assertTrue(
            np.allclose(
                angle_between(self.vecs, v2),
                np.array([
                    1.37340077, 1.3328552, 1.3002466, 1.30024656, 1.3328552
                ])
            )
        )
        self.assertTrue(
            np.allclose(
                angle_between(self.vecs, v3),
                np.array([
                    1.57079633, 1.80873745, 2.13473897, 2.50107034, 2.80175574
                ])
            )
        )

    def test_angle_between_returns_nan_for_zero_length_vectors(self):
        v1 = (0, 0, 0)
        v2 = (1, 0, 0)
        with pytest.warns(RuntimeWarning):
            self.assertTrue(np.isnan(angle_between(v1, v2)))

    def test_space_angle(self):
        p1 = (np.pi / 2, np.pi)
        p2 = (np.pi, 0)
        self.assertAlmostEqual(
            space_angle(p1[0], p2[0], p1[1], p2[1]), 1.57079632679489
        )
        p3 = (0, np.pi)
        p4 = (np.pi / 2, 0)
        self.assertAlmostEqual(
            space_angle(p3[0], p4[0], p3[1], p4[1]), 1.57079632679489
        )

    def test_hsin(self):
        assert np.all(hsin((np.pi, 0)) == (1, 0))
        self.assertAlmostEqual(hsin(np.pi / 2), 0.5)

    def test_pld3(self):
        p1 = np.array((0, 0, 0))
        p2 = np.array((0, 0, 1))
        d2 = np.array((0, 1, 0))
        self.assertAlmostEqual(1, pld3(p1, p2, d2))
        p1 = np.array((0, 0, 0))
        p2 = np.array((0, 0, 2))
        d2 = np.array((0, 1, 0))
        self.assertAlmostEqual(2, pld3(p1, p2, d2))
        p1 = np.array((0, 0, 0))
        p2 = np.array((0, 0, 0))
        d2 = np.array((0, 1, 0))
        self.assertAlmostEqual(0, pld3(p1, p2, d2))
        p1 = np.array((1, 2, 3))
        p2 = np.array((4, 5, 6))
        d2 = np.array((7, 8, 9))
        self.assertAlmostEqual(0.5275893, pld3(p1, p2, d2))
        p1 = np.array((0, 0, 2))
        p2 = np.array((-100, 0, -100))
        d2 = np.array((1, 0, 1))
        self.assertAlmostEqual(1.4142136, pld3(p1, p2, d2))
        p1 = np.array([183., -311., 351.96083871])
        p2 = np.array([40.256, -639.888, 921.93])
        d2 = np.array([0.185998, 0.476123, -0.859483])
        self.assertAlmostEqual(21.25456308, pld3(p1, p2, d2))

    def test_com(self):
        center_of_mass = com(((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)))
        self.assertEqual((5.5, 6.5, 7.5), tuple(center_of_mass))
        center_of_mass = com(((1, 2, 3), (4, 5, 6), (7, 8, 9)),
                             masses=(1, 0, 0))
        self.assertEqual((1, 2, 3), tuple(center_of_mass))
        center_of_mass = com(((1, 1, 1), (0, 0, 0)))
        self.assertEqual((0.5, 0.5, 0.5), tuple(center_of_mass))


class TestShapes(TestCase):
    def setUp(self):
        self.poly = [
            (-60, 120),
            (80, 120),
            (110, 60),
            (110, -30),
            (70, -110),
            (-70, -110),
            (-90, -70),
            (-90, 60),
        ]

    def test_poly_containment(self):
        polygon = Polygon(self.poly)
        point_in = (-40, -40)
        point_out = (-140, -140)
        points = [
            (-40, -40),
            (-140, -140),
            (40, -140),
        ]
        assert np.all(polygon.contains(point_in))
        assert not np.any(polygon.contains(point_out))
        assert np.all(polygon.contains(points) == [True, False, False])

    def test_poly_xy(self):
        polygon = Polygon(self.poly)
        x = (-40, -140, 40)
        y = (-40, -140, -140)
        assert np.all(polygon.contains_xy(x, y) == [True, False, False])

    def test_prism_contained(self):
        z = (-90, 90)
        prism = IrregularPrism(self.poly, z[0], z[1])
        points = [
            (0, 1, 2),
            (-100, 20, 10),
            (10, 90, 10),
        ]
        assert np.all(prism.contains(points) == [True, False, True])

    def test_prism_contained_xyz(self):
        z = (-90, 90)
        prism = IrregularPrism(self.poly, z[0], z[1])
        x = (0, -100, 10)
        y = (1, 20, 90)
        z = (2, 10, 10)
        assert np.all(prism.contains_xyz(x, y, z) == [True, False, True])


class TestRotation(TestCase):
    def test_rotmat(self):
        v = [3, 5, 0]
        axis = [4, 4, 1]
        theta = 1.2
        newvec = np.dot(rotation_matrix(axis, theta), v)
        self.assertTrue(
            np.allclose(
                newvec, np.array([2.74911638, 4.77180932, 1.91629719])
            )
        )

    def test_cone(self):
        spike = [1, 1, 0]
        bottom = [0, 2, 0]
        angle = np.pi / 4
        n_angles = 20
        cone = SparseCone(spike, bottom, angle)
        circ_samp = cone.sample_circle(n_angles=n_angles)
        axis_samp = cone.sample_axis
        samp = cone.sample(n_angles)
        assert len(circ_samp) == n_angles
        assert len(axis_samp) == 2
        assert len(samp) == len(circ_samp) + 2


class TestLog(TestCase):
    def test_val(self):
        assert_allclose(log_b(5, 2), np.log2(5))
        assert_allclose(log_b(5, 10), np.log10(5))
        assert_allclose(log_b(5, np.e), np.log(5))


class TestQeuler(TestCase):
    def test_conversion_of_yaw(self):
        assert np.allclose([1, 0, 0, 0], qeuler(0, 0, 0))
        assert np.allclose([0.7071, 0, 0, 0.7071], qeuler(90, 0, 0))
        assert np.allclose([0, 0, 0, 1], qeuler(180, 0, 0))
        assert np.allclose([-0.7071, 0, 0, 0.7071], qeuler(270, 0, 0))
        assert np.allclose([-1, 0, 0, 0], qeuler(360, 0, 0))

    def test_conversion_of_pitch(self):
        assert np.allclose([0.92388, 0, 0.38268, 0], qeuler(0, 45, 0))
        assert np.allclose([0.92388, 0, -0.38268, 0], qeuler(0, -45, 0))
        assert np.allclose([0.7071, 0, 0.7071, 0], qeuler(0, 90, 0))
        assert np.allclose([0.8660254, 0, 0.5, 0], qeuler(0, 60, 0))
        assert np.allclose([-0.96592583, 0, -0.25881905, 0], qeuler(0, 390, 0))

    def test_conversion_of_roll(self):
        assert np.allclose([0.92388, 0.38268, 0, 0], qeuler(0, 0, 45))
        assert np.allclose([0.92388, -0.38268, 0, 0], qeuler(0, 0, -45))
        assert np.allclose([0.70710, 0.70710, 0, 0], qeuler(0, 0, 90))
        assert np.allclose([0.86602, 0.5, 0, 0], qeuler(0, 0, 60))
        assert np.allclose([-0.96592583, -0.25881905, 0, 0], qeuler(0, 0, 390))

    def test_mixed_conversion(self):
        assert np.allclose([0.999471, 0.02601972, 0.01767416, 0.00826538],
                           qeuler(1, 2, 3))
        assert np.allclose([0.94371436, 0.26853582, -0.14487813, 0.12767944],
                           qeuler(10, -20, 30))
        assert np.allclose([-0.16575384, -0.69624819, 0.05479592, -0.69624819],
                           qeuler(-999, 999, -999))


class TestQrot(TestCase):
    def test_rotation_of_x_vector(self):
        assert np.allclose([0, 1, 0], qrot([1, 0, 0], qeuler(90, 0, 0)))
        assert np.allclose([-1, 0, 0], qrot([1, 0, 0], qeuler(180, 0, 0)))
        assert np.allclose([-1, 0, 0], qrot([1, 0, 0], qeuler(180, 0, -45)))
        assert np.allclose([0, 0, -1], qrot([1, 0, 0], qeuler(180, 90, 45)))

    def test_rotation_of_y_vector(self):
        assert np.allclose([-1, 0, 0], qrot([0, 1, 0], qeuler(90, 0, 0)))
        assert np.allclose([0, -1, 0], qrot([0, 1, 0], qeuler(180, 0, 0)))
        assert np.allclose([0, -0.70710, -0.70710],
                           qrot([0, 1, 0], qeuler(180, 0, -45)))
        assert np.allclose([-0.70710, -0.70710, 0],
                           qrot([0, 1, 0], qeuler(180, 90, 45)))

    def test_rotation_of_z_vector(self):
        assert np.allclose([0, 0, 1], qrot([0, 0, 1], qeuler(90, 0, 0)))
        assert np.allclose([0, 0, 1], qrot([0, 0, 1], qeuler(180, 0, 0)))
        assert np.allclose([0, -0.70710, 0.70710],
                           qrot([0, 0, 1], qeuler(180, 0, -45)))
        assert np.allclose([-0.70710, 0.70710, 0],
                           qrot([0, 0, 1], qeuler(180, 90, 45)))

    def test_mixed_rotation(self):
        assert np.allclose([1, 2, 3], qrot([1, 2, 3], qeuler(0, 0, 0)))
        assert np.allclose([0, -1.414213, 0],
                           qrot([0, 1, -1], qeuler(180, 90, 45)))
        assert np.allclose([-1.41421356, 0, -1],
                           qrot([1, 1, 1], qeuler(180, 90, 45)))
        assert np.allclose([-14.1421356, 0, -10],
                           qrot([10, 10, 10], qeuler(180, 90, 45)))


class TestQrotYaw(TestCase):
    def test_call_with_list(self):
        qrot_yaw([1, 2, 3], 1)

    def test_no_rotation(self):
        vec = (1, 0, 0)
        vec_rot = qrot_yaw(vec, 0)
        assert np.allclose([1, 0, 0], vec_rot)

    def test_a_rotation_of_90(self):
        vec = (1, 0, 0)
        vec_rot = qrot_yaw(vec, 90)
        assert np.allclose([0, 1, 0], vec_rot)

    def test_a_rotation_of_180(self):
        vec = (1, 0, 0)
        vec_rot = qrot_yaw(vec, 180)
        assert np.allclose([-1, 0, 0], vec_rot)

    def test_a_full_rotation(self):
        vec = (1, 0, 0)
        vec_rot = qrot_yaw(vec, 360)
        assert np.allclose([1, 0, 0], vec_rot)

    def test_a_rotation_of_45(self):
        vec = (1, 0, 0)
        vec_rot = qrot_yaw(vec, 45)
        assert np.allclose([0.7071, 0.7071, 0], vec_rot)


class TestIntersect3D(TestCase):
    def test_intersection_at_zero(self):
        p1 = np.array([(1, 0, 0), (0, 0, 1)])
        p2 = -p1
        intersection = intersect_3d(p1, p2)
        assert np.allclose([0, 0, 0], intersection)

    def test_intersection_of_multiple_lines_with_same_endpoints(self):
        p1 = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
        p2 = np.array([(4, 4, 4), (4, 4, 4), (4, 4, 4)])
        intersection = intersect_3d(p1, p2)
        assert np.allclose([4, 4, 4], intersection)

    def test_intersection_of_multiple_lines_with_target(self):
        p1 = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
        target = np.array([23, 5, 42])
        p2 = 2 * target - p1
        intersection = intersect_3d(p1, p2)
        assert np.allclose(target, intersection)

    def test_another_intersection(self):
        p1 = np.array([(1, 10, 0), (0, 10, 1)])
        p2 = np.array([(-1, 10, 0), (0, 10, -1)])
        intersection = intersect_3d(p1, p2)
        assert np.allclose([0, 10, 0], intersection)


class TestDist(TestCase):
    def test_dist_between_two_2D_points(self):
        self.assertAlmostEqual(1, dist(np.array([0, 0]), np.array([1, 0])))
        self.assertAlmostEqual(
            np.sqrt(2), dist(np.array([0, 1]), np.array([1, 0]))
        )
        self.assertAlmostEqual(
            2 * np.sqrt(2), dist(np.array([1, 2]), np.array([3, 4]))
        )

    def test_dist_between_two_3D_points(self):
        self.assertAlmostEqual(
            1, dist(np.array([0, 0, 0]), np.array([1, 0, 0]))
        )
        self.assertAlmostEqual(
            np.sqrt(2), dist(np.array([0, 1, 0]), np.array([1, 0, 0]))
        )
        self.assertAlmostEqual(
            2, dist(np.array([0, 0, 2]), np.array([0, 0, 0]))
        )
        self.assertAlmostEqual(
            5.1961524, dist(np.array([1, 2, 3]), np.array([4, 5, 6]))
        )

    def test_dist_to_many_points(self):
        assert np.allclose([1, 1, 0, 1.73205081],
                           dist(
                               np.array([0, 0, 0]),
                               np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0],
                                         [1, 1, 1]]),
                               axis=1
                           ))
