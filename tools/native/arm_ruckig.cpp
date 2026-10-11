// Stateless batch calculation: caller commits only the collision-checked state.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <ruckig/ruckig.hpp>
namespace py = pybind11;
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::tuple calculate(Array state, Array goals, Array times, double vmax, double amax,
                    double jmax, bool brake, int threads) {
    if (state.ndim()!=3 || state.shape(1)!=3 || state.shape(2)!=14 ||
        goals.ndim()!=2 || goals.shape(0)!=state.shape(0) || goals.shape(1)!=14 ||
        times.ndim()!=1 || times.shape(0)<1 || vmax<=0 || amax<=0 || jmax<=0)
        throw std::invalid_argument("Expected state [N,3,14], goal [N,14], positive limits and times");
    const ssize_t n=state.shape(0), h=times.shape(0);
    Array result({n,h,ssize_t(3),ssize_t(14)}), extrema({n,ssize_t(14),ssize_t(2)});
    py::array_t<int> statuses(n);
    auto s=state.unchecked<3>(); auto g=goals.unchecked<2>(); auto ts=times.unchecked<1>();
    auto out=result.mutable_unchecked<4>(); auto ext=extrema.mutable_unchecked<3>();
    auto status=statuses.mutable_unchecked<1>();
    for(ssize_t k=0;k<h;++k) if(ts(k)<=0) throw std::invalid_argument("Times must be positive");
    {
      py::gil_scoped_release release;
      #pragma omp parallel for num_threads(threads) schedule(static) if(n>32)
      for(ssize_t i=0;i<n;++i) {
        ruckig::Ruckig<14> otg;
        ruckig::InputParameter<14> in;
        ruckig::Trajectory<14> tr;
        // Independent synchronization avoids artificial coupling of a tiny
        // waist correction to a nearly stationary joint. All axes stay bounded.
        in.synchronization=ruckig::Synchronization::None;
        if(brake) in.control_interface=ruckig::ControlInterface::Velocity;
        for(int j=0;j<14;++j) {
          in.current_position[j]=s(i,0,j);in.current_velocity[j]=s(i,1,j);in.current_acceleration[j]=s(i,2,j);
          in.target_position[j]=g(i,j);in.target_velocity[j]=0;in.target_acceleration[j]=0;
          in.max_velocity[j]=vmax;in.max_acceleration[j]=amax;in.max_jerk[j]=jmax;
        }
        int code;
        try { code=static_cast<int>(otg.calculate(in,tr)); } catch(...) {code=-1000;}
        status(i)=code;
        if(code<0) {
          for(ssize_t k=0;k<h;++k) for(int d=0;d<3;++d) for(int j=0;j<14;++j) out(i,k,d,j)=s(i,d,j);
          for(int j=0;j<14;++j) ext(i,j,0)=ext(i,j,1)=s(i,0,j);
          continue;
        }
        auto bounds=tr.get_position_extrema();
        for(int j=0;j<14;++j) {ext(i,j,0)=bounds[j].min;ext(i,j,1)=bounds[j].max;}
        for(ssize_t k=0;k<h;++k) {
          std::array<double,14> p,v,a;
          tr.at_time(ts(k),p,v,a);
          for(int j=0;j<14;++j) {out(i,k,0,j)=p[j];out(i,k,1,j)=v[j];out(i,k,2,j)=a[j];}
        }
      }
    }
    return py::make_tuple(result,extrema,statuses);
}
PYBIND11_MODULE(_arm_ruckig,m) {m.def("calculate", &calculate);}
