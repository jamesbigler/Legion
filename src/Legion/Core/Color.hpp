
#ifndef LEGION_CORE_COLOR_HPP_
#define LEGION_CORE_COLOR_HPP_

namespace legion
{

    class Color
    {
    public:
        Color() {}
        explicit Color( float rgb );
        Color( float r, float g, float b );

        float red()const           { return m_c[0]; }
        float green()const         { return m_c[1]; }
        float blue()const          { return m_c[2]; }

        void  setRed( float r )    { m_c[0] = r; }
        void  setGreen( float g )  { m_c[1] = g; }
        void  setBlue( float b )   { m_c[2] = b; }
    private:

        float m_c[3];

    };

        
    inline Color::Color( float rgb )
    {
        m_c[0] = m_c[1] = m_c[2] = rgb;
    }


    inline Color::Color( float r, float g, float b )
    {
        m_c[0] = r;
        m_c[1] = g;
        m_c[2] = b;
    }


}

#endif //  LEGION_CORE_COLOR_HPP_
