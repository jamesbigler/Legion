
#ifndef LEGION_SHADER_H_
#define LEGION_SHADER_H_


namespace legion
{

class IShader
{
public:
    Shader( const std::string& name );
    virtual ~Shader();

    virtual void  sample  ( const Vector3& w_out, const SurfaceGeometry& p, Vector3& w_in, float& pdf )=0;
    virtual float pdf     ( const Vector3& w_out, const SurfaceGeometry& p, const Vector3& w_in )=0;
    virtual Color evaluate( const Vector3& w_out, const SurfaceGeometry& p, const Vector3& w_in )=0;
private:
};

}
#endif // LEGION_SHADER_H_
