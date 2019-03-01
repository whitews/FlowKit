#include <stdio.h>
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