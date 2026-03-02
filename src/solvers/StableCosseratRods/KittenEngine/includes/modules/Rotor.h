#pragma once
namespace Kitten {
// quaternion
template <typename T>
struct RotorX {
    static constexpr double r2d = 57.2957795130823208767981548141051703324;
    static constexpr double myPi = 3.141592653589793238462643;
    static constexpr double my2Pi = 6.2831853071795864769252867665590057683943;

    KITTEN_FUNC_DECL static double my_fmod(double x, double y) {
        double ans;

        double lx = x;
        double ly = y;

        if (ly != 0) {
            ans = lx - (long long int)(lx / ly) * ly;
        } else {
            ans = 0;
        }

        return ans;
    }

    KITTEN_FUNC_DECL static double fitAngle(double angle, bool test = false) {
        if (test) printf("angle: %.10lf -> ", angle * r2d);
        angle = my_fmod(my_fmod(angle + myPi, my2Pi) + my2Pi, my2Pi) - myPi;
        if (test) printf("%.10lf\n\n", angle * r2d);
        return angle;
    }

    // 3d vec that used as an Euler angle
    typedef glm::vec<3, T, glm::defaultp> EulerAngleKT;
    // 3d vec that used as an axis
    typedef glm::vec<3, T, glm::defaultp> UnitAxisVecKT;
    // 4d vec that used as a quaternion
    typedef glm::vec<4, T, glm::defaultp> QuatKT;

    union {
        QuatKT v;
        struct {
            EulerAngleKT q;  // The bivector part laid out in the { y^z, z^x, x^y } basis
            T w;             // The scaler part
        };
        struct {
            T x, y, z, s;
        };
    };

    // Create a quaternion from an angle (in radians) and rotation axis
    KITTEN_FUNC_DECL static RotorX<T> angleAxis(T rad, UnitAxisVecKT axis) {
        rad *= 0.5;
        return RotorX<T>(sin(rad) * axis, cos(rad));
    }

    // Create a quaternion from an angle (in degrees) and rotation axis
    KITTEN_FUNC_DECL static RotorX<T> angleAxisDeg(T deg, UnitAxisVecKT axis) {
        deg *= 0.00872664625997165;
        return RotorX<T>(sin(deg) * axis, cos(deg));
    }

    // Create a quaternion from euler angles in radians
    KITTEN_FUNC_DECL static RotorX<T> eulerAngles(EulerAngleKT rad) {
        rad *= 0.5;
        EulerAngleKT c = cos(rad);
        EulerAngleKT s = sin(rad);
        return RotorX<T>(EulerAngleKT(0, 0, s.z), c.z) * RotorX<T>(EulerAngleKT(0, s.y, 0), c.y) * RotorX<T>(EulerAngleKT(s.x, 0, 0), c.x);
    }

    // Create a quaternion from euler angles in radians
    KITTEN_FUNC_DECL static RotorX<T> eulerAngles(T x, T y, T z) { return eulerAngles(EulerAngleKT(x, y, z)); }

    // Create a quaternion from euler angles in degrees
    KITTEN_FUNC_DECL static RotorX<T> eulerAnglesDeg(EulerAngleKT deg) { return eulerAngles(deg * 0.0174532925199432958f); }

    // Create a quaternion from euler angles in degrees
    KITTEN_FUNC_DECL static RotorX<T> eulerAnglesDeg(T x, T y, T z) { return eulerAnglesDeg(EulerAngleKT(x, y, z)); }

    KITTEN_FUNC_DECL static RotorX<T> fromTo(EulerAngleKT from, EulerAngleKT to) {
        EulerAngleKT h = (from + to) * 0.5f;
        T l = length2(h);
        if (l > 0)
            h *= inversesqrt(l);
        else
            h = orthoBasisX((vec3)from)[1];

        return RotorX<T>(cross(from, h), dot(from, h));
    }

    // Returns the multiplicative identity rotor
    KITTEN_FUNC_DECL static RotorX<T> identity() { return RotorX<T>(); }

    KITTEN_FUNC_DECL RotorX(T x, T y = 0, T z = 0, T w = 0) : v(x, y, z, w) {}
    KITTEN_FUNC_DECL RotorX(EulerAngleKT q, T w = 0) : q(q), w(w) {}
    KITTEN_FUNC_DECL RotorX(QuatKT v) : v(v) {}
    KITTEN_FUNC_DECL RotorX() : v(0, 0, 0, 1) {}

    KITTEN_FUNC_DECL RotorX(const RotorX<T>& other) : v(other.v) {}

    template <typename U>
    KITTEN_FUNC_DECL explicit RotorX(const RotorX<U>& other) : v((QuatKT)other.v) {}

    KITTEN_FUNC_DECL RotorX<T>& operator=(const RotorX<T>& rhs) {
        v = rhs.v;
        return *this;
    }

    // Get the multiplicative inverse
    KITTEN_FUNC_DECL RotorX<T> inverse() const { return RotorX<T>(-q, w); }

    KITTEN_FUNC_DECL RotorX<T> operator-() const { return inverse(); }

    // Rotate a vector by this rotor
    KITTEN_FUNC_DECL EulerAngleKT rotate(EulerAngleKT v) const {
        // Calculate v * ab
        EulerAngleKT a = w * v + cross(q, v);  // The vector
        T c = dot(v, q);                       // The trivector

        // Calculate (w - q) * (a + c). Ignoring the scaler-trivector parts
        return w * a          // The scaler-vector product
               + cross(q, a)  // The bivector-vector product
               + c * q;       // The bivector-trivector product
    }

    KITTEN_FUNC_DECL mat<3, 3, T, defaultp> matrix() {
        mat3 cm = crossMatrix(q);
        return abT(q, q) + mat<3, 3, T, defaultp>(w * w) + 2 * w * cm + cm * cm;
    }

    KITTEN_FUNC_DECL static RotorX<T> fromMatrix(mat<3, 3, T, defaultp> m) {
        RotorX<T> q;
        T t;
        if (m[2][2] < 0) {
            if (m[0][0] > m[1][1]) {
                t = 1 + m[0][0] - m[1][1] - m[2][2];
                q = RotorX<T>(t, m[1][0] + m[0][1], m[0][2] + m[2][0], m[2][1] - m[1][2]);
            } else {
                t = 1 - m[0][0] + m[1][1] - m[2][2];
                q = RotorX<T>(m[1][0] + m[0][1], t, m[2][1] + m[1][2], m[0][2] - m[2][0]);
            }
        } else {
            if (m[0][0] < -m[1][1]) {
                t = 1 - m[0][0] - m[1][1] + m[2][2];
                q = RotorX<T>(m[0][2] + m[2][0], m[2][1] + m[1][2], t, m[1][0] - m[0][1]);
            } else {
                t = 1 + m[0][0] + m[1][1] + m[2][2];
                q = RotorX<T>(m[2][1] - m[1][2], m[0][2] - m[2][0], m[1][0] - m[0][1], t);
            }
        }

        return RotorX<T>(((0.5f / glm::sqrt(t)) * q.v)).inverse();
    }

    // Get the euler angle in radians
    KITTEN_FUNC_DECL EulerAngleKT euler() const {
        return EulerAngleKT(atan2(2 * (w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y)), asin(2 * (w * q.y - q.x * q.z)),
                            atan2(2 * (w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)));
    }

    // Get the euler angle in degrees
    KITTEN_FUNC_DECL EulerAngleKT eulerDeg() const { return euler() * (T)r2d; }

    // Returns both the axis and rotation angle in radians
    KITTEN_FUNC_DECL UnitAxisVecKT axis(T& angle, bool test = false) const {
        T l = length(q);
        if (l == 0) {
            angle = 0;
            return UnitAxisVecKT(1, 0, 0);
        }

        angle = 2 * atan2(l, w);

        angle = fitAngle(angle);

        return q / l;
    }

    // Returns the axis of rotation
    KITTEN_FUNC_DECL UnitAxisVecKT axis(bool test = false) const {
        T a;
        return axis(a, test);
    }

    // Returns both the axis and rotation angle in degrees
    KITTEN_FUNC_DECL UnitAxisVecKT axisDeg(T& angle) {
        EulerAngleKT a = axis(angle);
        angle *= r2d;
        return a;
    }

    // Returns the angle of rotation in radians
    KITTEN_FUNC_DECL T angle(bool test = false) const {
        T a;
        axis(a, test);
        return a;
    }

    // Returns the angle of rotation in degrees
    KITTEN_FUNC_DECL T angleDeg() const {
        T a;
        axis(a);
        return a * r2d;
    }

    // Returns the unit vector that is along the Cosserat segment
    KITTEN_FUNC_DECL UnitAxisVecKT getD3() const { return glm::normalize(rotate(glm::vec3(1, 0, 0))); }

    KITTEN_FUNC_DECL friend EulerAngleKT operator*(RotorX<T> lhs, const EulerAngleKT& rhs) { return lhs.rotate(rhs); }

    KITTEN_FUNC_DECL friend RotorX<T> operator*(RotorX<T> lhs, const RotorX<T>& rhs) {
        return RotorX<T>(lhs.w * rhs.q + rhs.w * lhs.q + cross(lhs.q, rhs.q), lhs.w * rhs.w - dot(lhs.q, rhs.q));
    }

    KITTEN_FUNC_DECL RotorX<T>& operator+=(const RotorX<T>& rhs) {
        v += rhs.v;
        return *this;
    }

    KITTEN_FUNC_DECL friend RotorX<T> operator+(RotorX<T> lhs, const RotorX<T>& rhs) { return RotorX<T>(lhs.v + rhs.v); }

    KITTEN_FUNC_DECL friend RotorX<T> operator*(T lhs, const RotorX<T>& rhs) {
        if (rhs.w == 1) return RotorX<T>();
        T na = lhs * acos(rhs.w);  // New angle
        T nw = cos(na);            // New cosine
        T s = sqrt((1 - nw * nw) / length2(rhs.q));
        if (fract(na * 0.1591549430918954) > 0.5) s = -s;
        return RotorX<T>(rhs.q * s, nw);
    }

    // Gets the vec4 repersentation laid out in { y^z, z^x, x^y, scaler }
    KITTEN_FUNC_DECL explicit operator QuatKT() const { return v; }

    KITTEN_FUNC_DECL T& operator[](std::size_t idx) { return v[idx]; }
};

// mix rotors a to b from t=[0, 1] (unclamped)
template <typename T>
KITTEN_FUNC_DECL inline RotorX<T> mix(RotorX<T> a, RotorX<T> b, T t) {
    return (t * (b * a.inverse())) * a;
}

template <typename T>
KITTEN_FUNC_DECL inline T dot(RotorX<T> a, RotorX<T> b) {
    return glm::dot(a.v, b.v);
}

template <typename T>
KITTEN_FUNC_DECL inline RotorX<T> normalize(RotorX<T> a) {
    return RotorX<T>(glm::normalize(a.v));
}

// Projects x onto the constraint q*e = d
template <typename T>
KITTEN_FUNC_DECL inline RotorX<T> projectRotor(RotorX<T> x, vec<3, T, glm::defaultp> e, vec<3, T, glm::defaultp> d) {
    auto q = RotorX<T>::fromTo(e, d);
    auto qp = q.inverse() * x;
    qp.y = qp.z = 0;
    return q * normalize(qp);
}

template <typename T>
KITTEN_FUNC_DECL void print(RotorX<T> v, const char* format = "%.4f") {
    printf("{");
    for (int i = 0; i < 4; i++) {
        printf(format, v[i]);
        if (i != 3) printf(", ");
    }
    if (abs(length2(v.v) - 1) < 1e-3) {
        printf("}, euler: {");
        auto a = v.eulerDeg();
        for (int i = 0; i < 3; i++) {
            printf(format, a[i]);
            if (i != 2) printf(", ");
        }
        printf("} deg\n");
    } else
        printf("}\n");
}

using Rotor = RotorX<float>;
using RotorD = RotorX<double>;
}  // namespace Kitten