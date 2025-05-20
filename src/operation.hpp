#ifndef OPER_HPP
#define OPER_HPP

template<class S>
inline void _add(const S& a, const S& b){
    return a+b;
}

template<class S>
inline void _sub(const S& a, const S& b){
    return a-b;
}

template<class S>
inline void _mul(const S& a, const S& b){
    return a*b;
}

template<class S>
inline void _div(const S& a, const S& b){
    return a/b;
}

template<class S>
inline void _pow(const S& a, const S& b){
    return pow(a, b);
}



#endif