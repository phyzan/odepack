#ifndef VIRTUAL_BASE_IMPL_HPP
#define VIRTUAL_BASE_IMPL_HPP

#include "VirtualBase.hpp"

namespace ode{

template<typename T, size_t N>
void advance_all(const std::vector<OdeSolver<T, N>*>& solvers, T t_goal, int threads, bool display_progress){
    
    const int num = (threads <= 0) ? omp_get_max_threads() : threads;
    int tot = 0;
    const int target = solvers.size();
    Clock clock;
    clock.start();
    #pragma omp parallel for schedule(dynamic) num_threads(num)
    for (OdeSolver<T, N>* s : solvers){
        s->advance_until(t_goal);
        #pragma omp critical
        {
            if (display_progress){
                show_progress(++tot, target, clock);
            }
        }
    }
    std::cout << std::endl << "Parallel integration completed in: " << clock.message() << std::endl;

}

} // namespace ode

#endif // VIRTUAL_BASE_IMPL_HPP