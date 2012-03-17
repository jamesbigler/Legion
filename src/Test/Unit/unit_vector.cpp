
#include <Legion/Common/Math/Vector.hpp>
#include <Legion/Common/Util/Stream.hpp>

#include <iostream>

using namespace legion;

int main( int argc, char** argv )
{
    /*
    Vector3 v0( 0, 1, 2 );
    Vector3 v1( v0 );
    Vector3 v2 = v0 + v1;
    Vector2 v3( 4, 5 ); 
    std::cerr << v3.y() << std::endl;
    */

    Vector2 v4( atoi(argv[1]), atoi( argv[2] ) ); 
    float f = v4.normalize();

    std::cerr << "v4: " << v4 << " " << f <<  std::endl;
    //std::cin >> v2;
    
}
