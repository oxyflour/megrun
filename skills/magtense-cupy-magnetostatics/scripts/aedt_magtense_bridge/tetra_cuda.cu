// Analytical surface-charge integral for a uniformly magnetized tetrahedron.
// Independently implemented from the triangle edge/solid-angle identity:
// integral_face (r-r')/|r-r'|^3 dS = sum_edges nu*log((Ra+Rb+l)/(Ra+Rb-l)) - n*Omega.
// K_ab = sum_faces integral_a * n_b / (4*pi), so H = K M.
// Geometry and all transcendental operations use double, also for float output.
// No fast-math, dipole replacement, spectrum clipping, or changed quadrature.

__device__ inline double3 sub3(double3 a, double3 b) {
    return make_double3(a.x-b.x,a.y-b.y,a.z-b.z);
}
__device__ inline double3 cross3(double3 a, double3 b) {
    return make_double3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);
}
__device__ inline double dot3(double3 a, double3 b) {
    return a.x*b.x+a.y*b.y+a.z*b.z;
}
__device__ inline double3 read3(const double* a) {
    return make_double3(a[0],a[1],a[2]);
}
__device__ inline void write3(double* a, double3 v) {
    a[0]=v.x;a[1]=v.y;a[2]=v.z;
}

extern "C" __global__ void prepare_faces(const double* vertices, double* faces, int n) {
    int source=blockIdx.x*blockDim.x+threadIdx.x;
    if(source>=n) return;
    for(int opposite=0;opposite<4;opposite++) {
        double3 v[3];int k=0;
        for(int i=0;i<4;i++) if(i!=opposite) v[k++]=read3(vertices+12*source+3*i);
        double3 normal=cross3(sub3(v[1],v[0]),sub3(v[2],v[0]));
        if(dot3(normal,sub3(read3(vertices+12*source+3*opposite),v[0]))>0) {
            double3 tmp=v[1];v[1]=v[2];v[2]=tmp;
            normal=make_double3(-normal.x,-normal.y,-normal.z);
        }
        double norm=sqrt(dot3(normal,normal));
        normal=make_double3(normal.x/norm,normal.y/norm,normal.z/norm);
        double* f=faces+(source*4+opposite)*24;
        for(int i=0;i<3;i++) write3(f+3*i,v[i]);
        write3(f+9,normal);
        for(int i=0;i<3;i++) {
            double3 edge=sub3(v[(i+1)%3],v[i]);
            double length=sqrt(dot3(edge,edge));
            edge=make_double3(edge.x/length,edge.y/length,edge.z/length);
            write3(f+12+3*i,cross3(edge,normal));
            f[21+i]=length;
        }
    }
}

__device__ inline void point_tensor(const double* faces,int source,double3 point,double* out) {
    #pragma unroll
    for(int c=0;c<9;c++) out[c]=0.;
    for(int face=0;face<4;face++) {
        const double* f=faces+(source*4+face)*24;
        double3 v[3]={read3(f),read3(f+3),read3(f+6)};
        double3 r[3]={sub3(v[0],point),sub3(v[1],point),sub3(v[2],point)};
        double lengths[3]={sqrt(dot3(r[0],r[0])),sqrt(dot3(r[1],r[1])),sqrt(dot3(r[2],r[2]))};
        // Equivalent to r0 dot (r1 cross r2), avoiding cancellation far away.
        double numerator=dot3(r[0],cross3(sub3(v[1],v[0]),sub3(v[2],v[0])));
        double denominator=lengths[0]*lengths[1]*lengths[2]
             +dot3(r[0],r[1])*lengths[2]+dot3(r[1],r[2])*lengths[0]+dot3(r[2],r[0])*lengths[1];
        double omega=2.*atan2(numerator,denominator);
        double3 normal=read3(f+9);
        double field[3]={-omega*normal.x,-omega*normal.y,-omega*normal.z};
        for(int i=0;i<3;i++) {
            double length=f[21+i];
            double integral=log1p(2.*length/(lengths[i]+lengths[(i+1)%3]-length));
            #pragma unroll
            for(int a=0;a<3;a++) field[a]+=integral*f[12+3*i+a];
        }
        #pragma unroll
        for(int a=0;a<3;a++) {
            #pragma unroll
            for(int b=0;b<3;b++) out[3*a+b]+=field[a]*f[9+b];
        }
    }
    #pragma unroll
    for(int c=0;c<9;c++) out[c]*=0.079577471545947667884441881686257181;
}

template<typename T> __device__ inline void far_impl(
        const double* faces,const double* points,T* K,int n,int lo,int count) {
    long long pair=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(pair>=(long long)count*n) return;
    int source=pair%n, receiver=lo+pair/n;
    double sum[9]={0.};
    for(int q=0;q<4;q++) {
        double value[9];point_tensor(faces,source,read3(points+(receiver*4+q)*3),value);
        #pragma unroll
        for(int c=0;c<9;c++) sum[c]+=value[c];
    }
    #pragma unroll
    for(int a=0;a<3;a++) {
        #pragma unroll
        for(int b=0;b<3;b++) K[(long long)(receiver*3+a)*(3*n)+3*source+b]=(T)(sum[3*a+b]*.25);
    }
}

template<typename T> __device__ inline void near_impl(
        const double* faces,const double* points,const int* receivers,const int* sources,
        T* K,int n,int nq,int lo,int count) {
    int warp=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(warp>=count) return;
    int lane=threadIdx.x%32;
    int source=sources[lo+warp], receiver=receivers[lo+warp];
    double sum[9]={0.};
    for(int q=lane;q<nq;q+=32) {
        double value[9];point_tensor(faces,source,read3(points+((long long)receiver*nq+q)*3),value);
        #pragma unroll
        for(int c=0;c<9;c++) sum[c]+=value[c];
    }
    for(int offset=16;offset>0;offset/=2) {
        #pragma unroll
        for(int c=0;c<9;c++) sum[c]+=__shfl_down_sync(0xffffffff,sum[c],offset);
    }
    if(lane==0) {
        #pragma unroll
        for(int a=0;a<3;a++) {
            #pragma unroll
            for(int b=0;b<3;b++) K[(long long)(receiver*3+a)*(3*n)+3*source+b]=(T)(sum[3*a+b]/nq);
        }
    }
}

#define WRAPPERS(SUFFIX,TYPE) \
extern "C" __global__ void far_##SUFFIX(const double* f,const double* p,TYPE* K,int n,int lo,int count) { \
    far_impl(f,p,K,n,lo,count); } \
extern "C" __global__ void near_##SUFFIX(const double* f,const double* p,const int* r,const int* s,TYPE* K,int n,int nq,int lo,int count) { \
    near_impl(f,p,r,s,K,n,nq,lo,count); }
WRAPPERS(f64,double)
WRAPPERS(f32,float)
