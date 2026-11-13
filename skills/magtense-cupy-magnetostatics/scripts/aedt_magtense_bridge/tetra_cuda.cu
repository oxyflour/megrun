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

__device__ inline void face_field(const double* f,double3 point,double* field) {
        double3 v[3]={read3(f),read3(f+3),read3(f+6)};
        double3 r[3]={sub3(v[0],point),sub3(v[1],point),sub3(v[2],point)};
        double lengths[3]={sqrt(dot3(r[0],r[0])),sqrt(dot3(r[1],r[1])),sqrt(dot3(r[2],r[2]))};
        // Equivalent to r0 dot (r1 cross r2), avoiding cancellation far away.
        double numerator=dot3(r[0],cross3(sub3(v[1],v[0]),sub3(v[2],v[0])));
        double denominator=lengths[0]*lengths[1]*lengths[2]
             +dot3(r[0],r[1])*lengths[2]+dot3(r[1],r[2])*lengths[0]+dot3(r[2],r[0])*lengths[1];
        double omega=2.*atan2(numerator,denominator);
        double3 normal=read3(f+9);
        field[0]=-omega*normal.x;field[1]=-omega*normal.y;field[2]=-omega*normal.z;
        for(int i=0;i<3;i++) {
            double length=f[21+i];
            double integral=log1p(2.*length/(lengths[i]+lengths[(i+1)%3]-length));
            #pragma unroll
            for(int a=0;a<3;a++) field[a]+=integral*f[12+3*i+a];
        }
}

__device__ inline void point_tensor(const double* faces,int source,double3 point,double* out) {
    #pragma unroll
    for(int c=0;c<9;c++) out[c]=0.;
    for(int face=0;face<4;face++) {
        const double* f=faces+((long long)source*4+face)*24;
        double field[3];face_field(f,point,field);
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

// Compact analytical-minus-quadrature blocks for the matrix-free FMM path.
// Only a receiver batch of high-order quadrature is present, never N*256 points.
template<typename T> __device__ inline void fmm_blocks_impl(
        const double* faces,const double* source_points,const double* target_points,
        const double* weights,const double* high,const int* receivers,const int* sources,
        const int* use_high,int lo,int nsq,int nhq,int count,T* blocks,T* diagonal) {
    int pair=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(pair>=count) return;
    int lane=threadIdx.x%32, receiver=receivers[pair], source=sources[pair];
    int nq=use_high[pair]?nhq:4;
    const double* points=use_high[pair]?high+(long long)(receiver-lo)*nhq*3
                                      :target_points+(long long)receiver*12;
    double exact[9]={0.}, approximate[9]={0.};
    for(int q=lane;q<nq;q+=32) {
        double value[9];point_tensor(faces,source,read3(points+3*q),value);
        for(int c=0;c<9;c++) exact[c]+=value[c]/nq;
    }
    for(int q=lane;q<4*nsq;q+=32) {
        double3 r=sub3(read3(target_points+(long long)receiver*12+3*(q/nsq)),
                       read3(source_points+((long long)source*nsq+q%nsq)*3));
        double radius2=dot3(r,r);
        if(radius2==0.) continue; // Same exclusion as source-evaluated FMM3D.
        double inverse=1./sqrt(radius2);
        double r3=inverse*inverse*inverse, r5=r3*inverse*inverse;
        double coordinates[3]={r.x,r.y,r.z};
        for(int a=0;a<3;a++) for(int b=0;b<3;b++)
            approximate[3*a+b]+=(3*coordinates[a]*coordinates[b]*r5-(a==b?r3:0.))
                *weights[source]*0.019894367886486916971110470421564;
    }
    for(int offset=16;offset>0;offset/=2) for(int c=0;c<9;c++) {
        exact[c]+=__shfl_down_sync(0xffffffff,exact[c],offset);
        approximate[c]+=__shfl_down_sync(0xffffffff,approximate[c],offset);
    }
    if(lane==0) for(int c=0;c<9;c++) {
        blocks[(long long)pair*9+c]=(T)(exact[c]-approximate[c]);
        if(source==receiver) diagonal[(long long)receiver*9+c]=(T)exact[c];
    }
}

template<typename T> __device__ inline void fmm_apply_impl(
        const T* blocks,const int* sources,const long long* pointer,
        const T* vector,int n,T* result) {
    int receiver=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(receiver>=n) return;
    int lane=threadIdx.x%32;
    double sum[3]={0.};
    for(long long pair=pointer[receiver]+lane;pair<pointer[receiver+1];pair+=32) {
        long long source=sources[pair];
        for(int a=0;a<3;a++) for(int b=0;b<3;b++)
            sum[a]+=(double)blocks[pair*9+3*a+b]*vector[3*source+b];
    }
    for(int offset=16;offset>0;offset/=2) for(int a=0;a<3;a++)
        sum[a]+=__shfl_down_sync(0xffffffff,sum[a],offset);
    if(lane==0) for(int a=0;a<3;a++) result[(long long)receiver*3+a]=(T)sum[a];
}

#define FMM_WRAPPERS(SUFFIX,TYPE) \
extern "C" __global__ void fmm_blocks_##SUFFIX( \
    const double* f,const double* s,const double* t,const double* w,const double* h, \
    const int* r,const int* i,const int* u,int lo,int nsq,int nhq,int count,TYPE* b,TYPE* d) { \
    fmm_blocks_impl(f,s,t,w,h,r,i,u,lo,nsq,nhq,count,b,d); } \
extern "C" __global__ void fmm_apply_##SUFFIX( \
    const TYPE* b,const int* i,const long long* p,const TYPE* v,int n,TYPE* r) { \
    fmm_apply_impl(b,i,p,v,n,r); }
FMM_WRAPPERS(f64,double)
FMM_WRAPPERS(f32,float)

// Radius search in CuPy's left-balanced heap-layout KD tree. Unlike the
// generic query_ball_point wrapper, output is counted before compact CSR
// allocation; no receiver-count x source-count intermediate is created.
extern "C" __global__ void fmm_neighbours(
        const double* tree,const long long* order,const int* global_ids,
        const double* radii,const double* points,const double* receiver_radii,
        int n,int receivers,double maximum_radius,double factor,int fill,
        long long* counts,long long* cursor,unsigned long long* keys) {
    int receiver=blockIdx.x*blockDim.x+threadIdx.x;
    if(receiver>=receivers) return;
    const double* point=points+3*receiver;
    double rr=receiver_radii[receiver];
    double search=nextafter(factor*(rr+maximum_radius),__longlong_as_double(0x7ff0000000000000LL));
    int stack[64],top=0;stack[top++]=0;
    long long count=0;
    while(top) {
        int node=stack[--top];
        const double* source=tree+(long long)node*3;
        double dx=point[0]-source[0],dy=point[1]-source[1],dz=point[2]-source[2];
        int id=global_ids[order[node]];
        if(sqrt(dx*dx+dy*dy+dz*dz)<=factor*(rr+radii[id])) {
            if(fill) keys[cursor[receiver]+count]=((unsigned long long)receiver<<32)|(unsigned int)id;
            count++;
        }
        int axis=(31-__clz((unsigned int)node+1))%3;
        double delta=point[axis]-source[axis];
        long long left=(long long)node*2+1,right=left+1;
        long long close=delta<=0?left:right,far=delta<=0?right:left;
        if(far<n && fabs(delta)<=search) stack[top++]=(int)far;
        if(close<n) stack[top++]=(int)close;
    }
    if(fill) cursor[receiver]+=count;
    else counts[receiver]+=count;
}

// Interpolation FMM: compact Morton octrees, polynomial moments/local values,
// exact point-dipole neighbours. All arrays and floating-point work stay on GPU.
__device__ unsigned long long morton3(int x,int y,int z) {
    unsigned long long key=0;
    for(int b=0;b<21;b++) {
        key|=((unsigned long long)((x>>b)&1))<<(3*b);
        key|=((unsigned long long)((y>>b)&1))<<(3*b+1);
        key|=((unsigned long long)((z>>b)&1))<<(3*b+2);
    }
    return key;
}
__device__ int3 unmorton3(unsigned long long key) {
    int3 p=make_int3(0,0,0);
    for(int b=0;b<21;b++) {
        p.x|=((key>>(3*b))&1)<<b;
        p.y|=((key>>(3*b+1))&1)<<b;
        p.z|=((key>>(3*b+2))&1)<<b;
    }
    return p;
}
extern "C" __global__ void fmm_morton(const double* points,int n,int depth,
                                     unsigned long long* keys) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    int side=1<<depth;
    int x=min(side-1,max(0,(int)floor(points[3LL*i]*side)));
    int y=min(side-1,max(0,(int)floor(points[3LL*i+1]*side)));
    int z=min(side-1,max(0,(int)floor(points[3LL*i+2]*side)));
    keys[i]=morton3(x,y,z);
}
extern "C" __global__ void fmm_box_lookup(
        const unsigned long long* targets,const unsigned long long* sources,
        int nt,int ns,int depth,int dx,int dy,int dz,int interaction,int* result) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nt) return;
    int3 t=unmorton3(targets[i]);
    int x=t.x-dx,y=t.y-dy,z=t.z-dz,side=1<<depth;
    result[i]=-1;
    if(x<0 || y<0 || z<0 || x>=side || y>=side || z>=side) return;
    if(interaction && (abs((t.x>>1)-(x>>1))>1 || abs((t.y>>1)-(y>>1))>1
                       || abs((t.z>>1)-(z>>1))>1)) return;
    unsigned long long key=morton3(x,y,z);
    int lo=0,hi=ns;
    while(lo<hi) {int mid=lo+(hi-lo)/2;if(sources[mid]<key) lo=mid+1;else hi=mid;}
    if(lo<ns && sources[lo]==key) result[i]=lo;
}
__device__ double fmm_basis(double x,int node,int p,const double* nodes,const double* weights) {
    double denominator=0.;
    for(int j=0;j<p;j++) {
        if(x==nodes[j]) return node==j?1.:0.;
        denominator+=weights[j]/(x-nodes[j]);
    }
    return (weights[node]/(x-nodes[node]))/denominator;
}
extern "C" __global__ void fmm_p2m(
        const double* points,const double* moments,const long long* order,
        const long long* pointer,const unsigned long long* keys,int first,int boxes,int depth,
        int p,const double* nodes,const double* weights,double* output) {
    int box=first+blockIdx.x;
    if(box>=boxes) return;
    int3 c=unmorton3(keys[box]);
    int side=1<<depth,k=p*p*p;
    for(int node=threadIdx.x;node<k;node+=blockDim.x) {
        int a=node/(p*p),b=(node/p)%p,d=node%p;
        double sum[3]={0.,0.,0.};
        for(long long j=pointer[box];j<pointer[box+1];j++) {
            long long i=order[j];const double* r=points+3*i;
            if(moments[3*i]==0. && moments[3*i+1]==0. && moments[3*i+2]==0.) continue;
            double w=fmm_basis(2*(r[0]*side-c.x)-1,a,p,nodes,weights)
                    *fmm_basis(2*(r[1]*side-c.y)-1,b,p,nodes,weights)
                    *fmm_basis(2*(r[2]*side-c.z)-1,d,p,nodes,weights);
            for(int axis=0;axis<3;axis++) sum[axis]+=w*moments[3*i+axis];
        }
        for(int axis=0;axis<3;axis++) output[((long long)box*k+node)*3+axis]=sum[axis];
    }
}
extern "C" __global__ void fmm_l2p(
        const double* points,const long long* order,const int* boxes,
        const unsigned long long* keys,const double* local,int first,int stop,int depth,
        int p,const double* nodes,const double* weights,double* output) {
    int j=first+(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(j>=stop) return;
    int lane=threadIdx.x%32,box=boxes[j],side=1<<depth,k=p*p*p;
    long long i=order[j];int3 c=unmorton3(keys[box]);const double* r=points+3*i;
    double sum[3]={0.,0.,0.};
    for(int node=lane;node<k;node+=32) {
        double w=fmm_basis(2*(r[0]*side-c.x)-1,node/(p*p),p,nodes,weights)
                *fmm_basis(2*(r[1]*side-c.y)-1,(node/p)%p,p,nodes,weights)
                *fmm_basis(2*(r[2]*side-c.z)-1,node%p,p,nodes,weights);
        for(int axis=0;axis<3;axis++) sum[axis]+=w*local[((long long)box*k+node)*3+axis];
    }
    for(int d=16;d>0;d/=2) for(int a=0;a<3;a++) sum[a]+=__shfl_down_sync(0xffffffff,sum[a],d);
    if(lane==0) for(int a=0;a<3;a++) output[3*i+a]=sum[a];
}
__device__ void dipole_add(const double* source,const double* target,const double* m,double* sum) {
    if(m[0]==0. && m[1]==0. && m[2]==0.) return;
    double x=target[0]-source[0],y=target[1]-source[1],z=target[2]-source[2];
    double r2=x*x+y*y+z*z;if(r2==0.) return;
    double inverse=1./sqrt(r2),r3=inverse*inverse*inverse;
    double dot=x*m[0]+y*m[1]+z*m[2],r5=3*dot*r3/r2;
    double r[3]={x,y,z};
    for(int a=0;a<3;a++) sum[a]+=(r[a]*r5-m[a]*r3)*0.079577471545947667884441881686257;
}
extern "C" __global__ void fmm_p2p(
        const double* sources,const double* targets,const double* moments,
        const long long* source_order,const long long* target_order,const long long* pointer,
        const int* boxes,const int* neighbours,int first,int stop,double* output) {
    int j=first+(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(j>=stop) return;
    int lane=threadIdx.x%32,box=boxes[j];long long i=target_order[j];
    double sum[3]={0.,0.,0.};
    for(int neighbour=0;neighbour<27;neighbour++) {
        int source_box=neighbours[(long long)box*27+neighbour];
        if(source_box<0) continue;
        for(long long q=pointer[source_box]+lane;q<pointer[source_box+1];q+=32) {
            long long s=source_order[q];
            dipole_add(sources+3*s,targets+3*i,moments+3*s,sum);
        }
    }
    for(int d=16;d>0;d/=2) for(int a=0;a<3;a++) sum[a]+=__shfl_down_sync(0xffffffff,sum[a],d);
    if(lane==0) for(int a=0;a<3;a++) output[3*i+a]+=sum[a];
}
extern "C" __global__ void fmm_point_reference(
        const double* sources,const double* targets,const double* moments,
        int first,int stop,int nt,double* output) {
    int i=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(i>=nt) return;
    int lane=threadIdx.x%32;double sum[3]={0.,0.,0.};
    for(int s=first+lane;s<stop;s+=32) dipole_add(sources+3LL*s,targets+3LL*i,moments+3LL*s,sum);
    for(int d=16;d>0;d/=2) for(int a=0;a<3;a++) sum[a]+=__shfl_down_sync(0xffffffff,sum[a],d);
    if(lane==0) for(int a=0;a<3;a++) output[3LL*i+a]+=sum[a];
}

extern "C" __global__ void fmm_air_correction(
        const double* faces,const double* points,const double* magnetization,
        const double* source_points,const double* weights,const int* indices,
        const long long* pointer,int receivers,int nq,int offset,double plane_tolerance,double* output) {
    int receiver=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(receiver>=receivers) return;
    int lane=threadIdx.x%32;const double* point=points+3*receiver;
    double sum[3]={0.,0.,0.};
    long long first=pointer[receiver]+offset,stop=min(pointer[receiver+1],first+8192);
    for(long long pair=first+lane;pair<stop;pair+=32) {
        int source=indices[pair];bool coplanar=false;
        for(int face=0;face<4;face++) {
            const double* f=faces+((long long)source*4+face)*24;
            if(fabs(dot3(read3(f+9),sub3(read3(point),read3(f))))<=plane_tolerance) coplanar=true;
        }
        if(coplanar) {sum[0]=sum[1]=sum[2]=__longlong_as_double(0x7ff8000000000000LL);break;}
        double tensor[9];point_tensor(faces,source,read3(point),tensor);
        const double* m=magnetization+3LL*source;
        for(int a=0;a<3;a++) for(int b=0;b<3;b++) sum[a]+=tensor[3*a+b]*m[b];
        double moment[3]={m[0]*weights[source],m[1]*weights[source],m[2]*weights[source]};
        double approximate[3]={0.,0.,0.};
        for(int q=0;q<nq;q++) dipole_add(source_points+((long long)source*nq+q)*3,point,moment,approximate);
        for(int a=0;a<3;a++) sum[a]-=approximate[a];
    }
    for(int d=16;d>0;d/=2) for(int a=0;a<3;a++) sum[a]+=__shfl_down_sync(0xffffffff,sum[a],d);
    if(lane==0) for(int a=0;a<3;a++) output[3*receiver+a]+=sum[a];
}

extern "C" __global__ void fmm_face_charges(const double* faces,const double* m,int n,double* charges) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=n) return;
    for(int face=0;face<4;face++) charges[4LL*i+face]=dot3(read3(faces+((long long)i*4+face)*24+9),read3(m+3LL*i));
}

// Packed fixed-near H-matrix: all right factors in one launch, then all left
// factors in one launch. Metadata is [row0,col0,z0,left0,right0,nrow,ncol,rank].
extern "C" __global__ void fmm_compressed_right(
        const double* right,const int* columns,const long long* meta,
        const int* rank_blocks,const double* vector,int count,double* intermediate) {
    int row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(row>=count) return;
    int lane=threadIdx.x%32,b=rank_blocks[row];
    const long long* d=meta+8LL*b;
    long long k=row-d[2],n=d[6];
    double value=0.;
    for(long long j=lane;j<n;j+=32) value+=right[d[4]+k*n+j]*vector[columns[d[1]+j]];
    for(int step=16;step>0;step/=2) value+=__shfl_down_sync(0xffffffff,value,step);
    if(lane==0) intermediate[row]=value;
}

extern "C" __global__ void fmm_compressed_left(
        const double* left,const int* rows,const long long* meta,
        const int* row_blocks,const double* intermediate,int count,double* output) {
    int row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(row>=count) return;
    int lane=threadIdx.x%32,b=row_blocks[row];
    const long long* d=meta+8LL*b;
    long long i=row-d[0],rank=d[7];
    double value=0.;
    for(long long k=lane;k<rank;k+=32) value+=left[d[3]+i*rank+k]*intermediate[d[2]+k];
    for(int step=16;step>0;step/=2) value+=__shfl_down_sync(0xffffffff,value,step);
    if(lane==0) atomicAdd(output+rows[row],value);
}

extern "C" __global__ void fmm_correct_vectors(
        const double* faces,const double* charges,const double* moments,
        const double* sources,const double* targets,const double* high,
        const double* centres,const double* radii,const double* weights,
        const int* receivers,const int* indices,const int* high_rows,int nq,int nhq,int count,
        double near_factor,double* result) {
    int pair=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(pair>=count) return;
    int lane=threadIdx.x%32,r=receivers[pair],s=indices[pair];
    if(moments[3LL*s]==0. && moments[3LL*s+1]==0. && moments[3LL*s+2]==0.) {
        if(lane==0) for(int a=0;a<3;a++) result[3LL*pair+a]=0.;
        return;
    }
    double3 delta=sub3(read3(centres+3LL*r),read3(centres+3LL*s));
    bool close=sqrt(dot3(delta,delta))<=near_factor*(radii[r]+radii[s]);
    int number=close?nhq:4;
    const double* points=close?high+(long long)high_rows[pair]*nhq*3:targets+12LL*r;
    double sum[3]={0.,0.,0.};
    for(int q=lane;q<number;q+=32) for(int face=0;face<4;face++) {
        double value[3];face_field(faces+((long long)s*4+face)*24,read3(points+3*q),value);
        double charge=charges[4LL*s+face]*0.079577471545947667884441881686257/number;
        for(int a=0;a<3;a++) sum[a]+=value[a]*charge;
    }
    double m[3]={moments[3LL*s]*weights[s]*.25,moments[3LL*s+1]*weights[s]*.25,
                 moments[3LL*s+2]*weights[s]*.25};
    for(int q=lane;q<4*nq;q+=32) {
        double approximate[3]={0.,0.,0.};
        dipole_add(sources+((long long)s*nq+q%nq)*3,targets+12LL*r+3*(q/nq),m,approximate);
        for(int a=0;a<3;a++) sum[a]-=approximate[a];
    }
    for(int d=16;d>0;d/=2) for(int a=0;a<3;a++) sum[a]+=__shfl_down_sync(0xffffffff,sum[a],d);
    if(lane==0) for(int a=0;a<3;a++) result[3LL*pair+a]=sum[a];
}
extern "C" __global__ void fmm_sum_vectors(const double* vectors,const long long* pointer,
                                          int n,double* result) {
    int r=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(r>=n) return;
    int lane=threadIdx.x%32;double sum[3]={0.,0.,0.};
    for(long long pair=pointer[r]+lane;pair<pointer[r+1];pair+=32)
        for(int a=0;a<3;a++) sum[a]+=vectors[3*pair+a];
    for(int d=16;d>0;d/=2) for(int a=0;a<3;a++) sum[a]+=__shfl_down_sync(0xffffffff,sum[a],d);
    if(lane==0) for(int a=0;a<3;a++) result[3*r+a]=sum[a];
}
// P0 single layer / P1 double layer for the scalar FEM--BEM boundary backend.
// Layout: vertices(9), normal(3), trace gradients(9), edge lengths(3), conormals(9).
__device__ inline void bem_source(const double* g,double3 x,double* values) {
    double3 r[3]={sub3(read3(g),x),sub3(read3(g+3),x),sub3(read3(g+6),x)};
    double R[3]={sqrt(dot3(r[0],r[0])),sqrt(dot3(r[1],r[1])),sqrt(dot3(r[2],r[2]))};
    double d=-dot3(r[0],read3(g+9));
    double denom=R[0]*R[1]*R[2]+dot3(r[0],r[1])*R[2]+dot3(r[1],r[2])*R[0]+dot3(r[2],r[0])*R[1];
    double omega=-2.*atan2(dot3(r[0],cross3(r[1],r[2])),denom);
    if(fabs(d)<7.105427357601002e-15*fmax(1.,sqrt(dot3(x,x)))) { d=0;omega=0; }
    double potential=-d*omega;double3 edge=make_double3(0,0,0);
    for(int j=0;j<3;j++) {
        double L=g[21+j];double v=log1p(2.*L/(R[j]+R[(j+1)%3]-L));
        double3 nu=read3(g+24+3*j);
        potential+=dot3(r[j],nu)*v;edge.x+=nu.x*v;edge.y+=nu.y*v;edge.z+=nu.z*v;
    }
    values[0]=potential/(4.*3.14159265358979323846);
    for(int j=0;j<3;j++) {
        double3 gradient=read3(g+12+3*j);
        double bary=(j==0 ? 1.:0.)-dot3(r[0],gradient);
        values[j+1]=-(bary*omega-d*dot3(edge,gradient))/(4.*3.14159265358979323846);
    }
}

__device__ inline void bem_pair(const double* target,const double* source,double area,
 const double* bary,const double* weights,int nq,double* sum) {
    for(int c=0;c<4;c++) sum[c]=0.;
    for(int p=0;p<nq;p++) {
        double3 x=make_double3(0,0,0);
        for(int v=0;v<3;v++) { x.x+=bary[3*p+v]*target[3*v];x.y+=bary[3*p+v]*target[3*v+1];x.z+=bary[3*p+v]*target[3*v+2]; }
        double value[4];bem_source(source,x,value);
        for(int c=0;c<4;c++) sum[c]+=weights[p]*value[c];
    }
    for(int c=0;c<4;c++) sum[c]*=area;
}

__device__ inline bool bem_coplanar(const double* target,const double* source) {
    bool planar=true;double3 origin=read3(source),normal=read3(source+9);
    for(int v=0;v<3;v++) {
        double3 x=read3(target+3*v);
        if(fabs(dot3(sub3(x,origin),normal))>=1.7763568394002505e-15*fmax(1.,sqrt(dot3(x,x)))) planar=false;
    }
    return planar;
}
__device__ inline double bem_entry(const double* geometry,const double* areas,const int* node_pointer,
 const int* node_triangles,const int* node_basis,const double* scales,int faces,int row,int column,
 const double* bary,const double* weights,int nq) {
    const double* target=geometry+33*(long long)row;double sum=0,value[4];
    if(column<faces) {
        bem_pair(target,geometry+33*(long long)column,areas[row],bary,weights,nq,value);
        sum=value[0];
    } else {
        int node=column-faces;
        for(int p=node_pointer[node];p<node_pointer[node+1];p++) {
            const double* source=geometry+33*(long long)node_triangles[p];
            // D(P1) is exactly zero for coplanar tests and a flat source.
            // This conservative shortcut is inside the existing pointwise
            // principal-value tolerance, including barycentric rounding.
            if(bem_coplanar(target,source)) continue;
            bem_pair(target,source,areas[row],bary,weights,nq,value);
            sum+=value[node_basis[p]+1];
        }
    }
    return sum*scales[column]/areas[row];
}

extern "C" __global__ void bem_coefficients(const double* geometry,const double* areas,
 const int* node_pointer,const int* node_triangles,const int* node_basis,const double* scales,int faces,
 const int* rows,const int* cols,int nr,int nc,const double* bary,const double* weights,int nq,double* output) {
    long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=(long long)nr*nc) return;
    int i=index/nc,j=index%nc;
    output[index]=bem_entry(geometry,areas,node_pointer,node_triangles,node_basis,scales,faces,
        rows[i],cols[j],bary,weights,nq);
}

// Modes: 0 variable-size exact blocks, 1 ACA pivot rows, 2 pivot columns,
// 3 fixed independent check crosses. All modes integrate the same panels.
extern "C" __global__ void bem_batch_coefficients(const double* geometry,const double* areas,
 const int* node_pointer,const int* node_triangles,const int* node_basis,const double* scales,int faces,
 const int* permutation,const int* columns,const long long* descriptors,int batches,int mode,const int* pivots,
 const int* check_rows,const int* check_columns,int pr,int pc,const long long* prefix,
 long long per_batch,long long total,const double* bary,const double* weights,int nq,
 const unsigned char* active,int skip_inactive,double* output) {
    long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(index>=total) return;
    int batch;long long local;
    if(mode==0 || mode==4) {
        int lo=0,hi=batches;
        while(lo+1<hi) { int mid=(lo+hi)/2;if(prefix[mid]<=index) lo=mid;else hi=mid; }
        batch=lo;local=index-prefix[lo];
    } else { batch=index/per_batch;local=index%per_batch; }
    if(skip_inactive && !active[batch]) { output[index]=0.;return; }
    const long long* a=descriptors+4*batch;int i,j;
    if(mode==0 || mode==4) { i=local/a[3];j=local%a[3]; }
    else if(mode==1) { i=pivots[batch];j=local; }
    else if(mode==2) { i=local;j=pivots[batch]; }
    else { i=check_rows[batch*pr+local/pc];j=check_columns[batch*pc+local%pc]; }
    int row=permutation[a[0]+i],column=(mode==4 ? permutation[a[2]+j]:columns[a[2]+j]);
    output[index]=bem_entry(geometry,areas,node_pointer,node_triangles,node_basis,scales,faces,
        row,column,bary,weights,nq);
}

// One target pivot / source triangle integral serves its P0 column and every
// incident P1 column. The incidence reduction below keeps the original order.
extern "C" __global__ void bem_pivot_panels(const double* geometry,const double* areas,
 const int* permutation,const long long* descriptors,const int* pivots,
 const long long* source_keys,const int* needs_single,int faces,long long count,
 const double* bary,const double* weights,int nq,const unsigned char* active,int skip_inactive,
 double* values,unsigned char* coplanar) {
    long long k=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(k>=count) return;
    long long key=source_keys[k];int batch=key/faces,source=key%faces;
    if(skip_inactive && !active[batch]) {
        for(int c=0;c<4;c++) values[4*k+c]=0.;coplanar[k]=1;return;
    }
    int row=permutation[descriptors[4LL*batch]+pivots[batch]];
    const double* target=geometry+33LL*row;const double* panel=geometry+33LL*source;
    bool planar=bem_coplanar(target,panel);coplanar[k]=planar;
    double result[4]={0.,0.,0.,0.};
    if(needs_single[k] || !planar) bem_pair(target,panel,areas[row],bary,weights,nq,result);
    for(int c=0;c<4;c++) values[4*k+c]=result[c];
}
extern "C" __global__ void bem_pivot_gather(const double* areas,const double* scales,
 const int* permutation,const long long* descriptors,const int* pivots,const int* columns,
 const long long* pointer,const int* panel_index,const int* component,const double* values,
 const unsigned char* coplanar,int nc,long long count,double* output) {
    long long k=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(k>=count) return;
    int batch=k/nc,row=permutation[descriptors[4LL*batch]+pivots[batch]];double sum=0.;
    for(long long p=pointer[k];p<pointer[k+1];p++) {
        int source=panel_index[p],c=component[p];
        if(c && coplanar[source]) continue;
        sum+=values[4LL*source+c];
    }
    output[k]=sum*scales[columns[k]]/areas[row];
}

__device__ inline double bem_coefficient(const void* data,long long index,int bytes) {
    return bytes==4 ? (double)((const float*)data)[index] : ((const double*)data)[index];
}
extern "C" __global__ void bem_h_forward(const long long* metadata,int count,int bytes,
 const double* row_areas,const double* x,double* y) {
    int b=blockIdx.x;if(b>=count) return;const long long* a=metadata+7*b;
    int rs=a[0],nr=a[1],cs=a[2],nc=a[3],rank=a[4];
    const void* left=(const void*)a[5];const void* right=(const void*)a[6];
    __shared__ double temp[128];
    if(rank<0) {
        for(int i=threadIdx.x;i<nr;i+=blockDim.x) {
            double sum=0;for(int j=0;j<nc;j++) sum+=bem_coefficient(left,(long long)i*nc+j,bytes)*x[cs+j];
            atomicAdd(y+rs+i,row_areas[rs+i]*sum);
        }
    } else {
        for(int k=threadIdx.x;k<rank;k+=blockDim.x) {
            double sum=0;for(int j=0;j<nc;j++) sum+=bem_coefficient(right,(long long)k*nc+j,bytes)*x[cs+j];temp[k]=sum;
        }
        __syncthreads();
        for(int i=threadIdx.x;i<nr;i+=blockDim.x) {
            double sum=0;for(int k=0;k<rank;k++) sum+=bem_coefficient(left,(long long)i*rank+k,bytes)*temp[k];atomicAdd(y+rs+i,row_areas[rs+i]*sum);
        }
    }
}

extern "C" __global__ void bem_h_transpose_v(const long long* metadata,int count,int bytes,
 const int* columns,const int* reverse,const double* scales,int faces,const double* q,double* y) {
    int b=blockIdx.x;if(b>=count) return;const long long* a=metadata+7*b;
    int rs=a[0],nr=a[1],cs=a[2],nc=a[3],rank=a[4];__shared__ double temp[128];
    const void* left=(const void*)a[5];const void* right=(const void*)a[6];
    if(rank<0) {
        for(int j=threadIdx.x;j<nc;j+=blockDim.x) {
            int column=columns[cs+j];if(column>=faces) continue;
            double sum=0;for(int i=0;i<nr;i++) sum+=bem_coefficient(left,(long long)i*nc+j,bytes)*q[rs+i];
            atomicAdd(y+reverse[column],.5*sum/scales[column]);
        }
    } else {
        for(int k=threadIdx.x;k<rank;k+=blockDim.x) {
            double sum=0;for(int i=0;i<nr;i++) sum+=bem_coefficient(left,(long long)i*rank+k,bytes)*q[rs+i];temp[k]=sum;
        }
        __syncthreads();
        for(int j=threadIdx.x;j<nc;j+=blockDim.x) {
            int column=columns[cs+j];if(column>=faces) continue;
            double sum=0;for(int k=0;k<rank;k++) sum+=bem_coefficient(right,(long long)k*nc+j,bytes)*temp[k];
            atomicAdd(y+reverse[column],.5*sum/scales[column]);
        }
    }
}

extern "C" __global__ void bem_h_inverse(const long long* metadata,const double* inverse,const double* x,double* y) {
    const long long* a=metadata+3*blockIdx.x;int start=a[0],n=a[1];
    for(int i=threadIdx.x;i<n;i+=blockDim.x) {
        double sum=0;for(int j=0;j<n;j++) sum+=inverse[a[2]+(long long)i*n+j]*x[start+j];y[start+i]=sum;
    }
}

extern "C" __global__ void bem_potential(const double* geometry,const double* vector,int faces,
 const double* points,double* output) {
    double3 x=read3(points+3*blockIdx.x);double sum=0;
    for(int f=threadIdx.x;f<faces;f+=blockDim.x) {
        double value[4];bem_source(geometry+33*(long long)f,x,value);
        for(int c=0;c<4;c++) sum-=value[c]*vector[4*(long long)f+c];
    }
    __shared__ double reduction[256];reduction[threadIdx.x]=sum;__syncthreads();
    for(int d=128;d;d/=2) { if(threadIdx.x<d) reduction[threadIdx.x]+=reduction[threadIdx.x+d];__syncthreads(); }
    if(threadIdx.x==0) output[blockIdx.x]=reduction[0];
}
