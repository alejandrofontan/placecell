/**
 * Scratch executable to test placecell during development.
 */
#include <iostream>

#include <placecell/placecell.h>

int main()
{
    placecell::PlaceCell place_cell;
    (void)place_cell;

    std::cout << "placecell skeleton OK (Eigen " << EIGEN_WORLD_VERSION << "."
              << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << ")" << std::endl;
    return 0;
}
