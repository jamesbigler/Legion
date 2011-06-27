
#include <Core/Vector.hpp>
#include <Util/Stream.hpp>
#include <iostream>

using namespace legion;

int main( int argc, char** argv )
{
    Vector3 v0( 0, 1, 2 );
    Vector3 v1( v0 );
    Vector3 v2 = v0 + v1;
    Vector2 v3( 4, 5 ); 
    std::cerr << v3.y() << std::endl;


    std::cerr << "v2: " << v2 << std::endl;
    //std::cin >> v2;
    
}
