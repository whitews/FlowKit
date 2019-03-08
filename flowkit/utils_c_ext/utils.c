#include <stdlib.h>
#include "utils.h"

double point_is_left(
        double point_a_x,
        double point_a_y,
        double point_b_x,
        double point_b_y,
        double test_point_x,
        double test_point_y
) {
    double is_left = (point_b_x - point_a_x) * (test_point_y - point_a_y) -
              (test_point_x - point_a_x) * (point_b_y - point_a_y);
    return is_left;
}

int calc_wind_count(double point_x, double point_y, int vert_count, double *poly_vertices) {
	int wind_count = 0;
	double vert_a_x;
	double vert_a_y;
	double vert_b_x;
	double vert_b_y;
	double is_left;

    // loop through all edges of the polygon
    for (int i=0; i<vert_count; i++) {
        //edge from poly_vertices[i] to poly_vertices[i+1]
        vert_a_x = poly_vertices[(i * 2) + 0];
        vert_a_y = poly_vertices[(i * 2) + 1];

        if (i >= vert_count - 1) {
            vert_b_x = poly_vertices[0];
            vert_b_y = poly_vertices[1];
        } else {
            vert_b_x = poly_vertices[(i * 2) + 2];
            vert_b_y = poly_vertices[(i * 2) + 3];
        }

        if (vert_a_y <= point_y) {
            if (point_y < vert_b_y) {
                // point crosses & edge travels upward
                is_left = point_is_left(vert_a_x, vert_a_y, vert_b_x, vert_b_y, point_x, point_y);
                if (is_left > 0) {
                    // point is left of edge
                    wind_count += 1;  // valid 'up' intersection
                }
            }
        } else {
            if (vert_b_y <= point_y) {
                // point crosses & edge travels downward
                is_left = point_is_left(vert_a_x, vert_a_y, vert_b_x, vert_b_y, point_x, point_y);

                if (is_left < 0) {
                    // point is right of edge
                    wind_count -= 1;  // valid 'down' intersect
                }
            }
        }
    }

    return wind_count;
}

int * points_in_polygon(double *poly_vertices, int vert_count, double *points, int point_count) {
    /*
    Determines whether points in an array are inside a polygon. Points on the
    edge of the polygon are considered inclusive. This function uses the
    winding number method and is robust to complex polygons with crossing
    boundaries, including the presence of 'holes' created by boundary crosses.

    This implementation is based on the C implementation here:

        http://geomalgorithms.com/a03-_inclusion.html

    Original copyright notice:
        Copyright 2000 softSurfer, 2012 Dan Sunday

    :param poly_vertices: Polygon vertices (array of 2-D points)
    :param vert_count: Number of vertices in polygon
    :param points: Points to test for polygon inclusion
    :param point_count: Number of points
    :return: Array of winding counts for each point. True is inside polygon.
    */
    int *wind_counts = malloc(point_count * sizeof(int));
    int wind_count;
    double point_x;
    double point_y;

    // First, find the polygon's bounding box & store the min/max values
    double min_x = poly_vertices[0];
    double max_x = poly_vertices[0];
    double min_y = poly_vertices[1];
    double max_y = poly_vertices[1];
    double vert_x, vert_y;
    
    for (int i=1; i<vert_count; i++) {
        vert_x = poly_vertices[(i * 2) + 0];
        vert_y = poly_vertices[(i * 2) + 1];
        
        if (vert_x < min_x) {
            min_x = vert_x;
        }
        else if (vert_x > max_x) {
            max_x = vert_x;
        }
        if (vert_y < min_y) {
            min_y = vert_y;
        }
        else if (vert_y > max_y) {
            max_y = vert_y;
        }
    }

    for (int i=0; i<point_count; i++) {
        point_x = points[i * 2];
        point_y = points[(i * 2) + 1];

        if (point_x < min_x || point_x > max_x || point_y < min_y || point_y > max_y) {
            wind_count = 0;
        } else {
            wind_count = calc_wind_count(
                point_x,
                point_y,
                vert_count,
                poly_vertices
            );
        }

        wind_counts[i] = wind_count;
    }

    return wind_counts;
}
