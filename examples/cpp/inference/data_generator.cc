#include "data_generator.h"
#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;

DataGenerator::DataGenerator(size_t _num_requests,
                             size_t _token_dim,
                             size_t _sequence_length,
                             bool _poisson_distr,
                             double _lambda)
    : num_requests(_num_requests), token_dim(_token_dim),
      sequence_length(_sequence_length), poisson_distr(_poisson_distr),
      lambda(_lambda), timer_started(false), global_unique_id(1000000) {
  generate_arrival_times();
};

void DataGenerator::generate_requests(float *req_ptr,
                                      int *label_ptr,
                                      int num_labels) {
  assert(req_ptr != nullptr);
  /* for (size_t i=0; i<num_requests; i++) {
    for (size_t j=0; j<sequence_length; j++) {
      for (size_t k=0; k<token_dim; k++) {
        req_ptr[i * sequence_length + j] = (float)std::rand()/RAND_MAX;
      }
    }
  } */
  random_device rnd_device;
  mt19937 mersenne_engine{rnd_device()};

  uniform_real_distribution<float> float_dist{0, 1.0};
  auto gen = [&float_dist, &mersenne_engine]() {
    return float_dist(mersenne_engine);
  };
  std::generate(
      req_ptr, req_ptr + token_dim * sequence_length * num_requests, gen);

  if (label_ptr != nullptr) {
    assert(num_labels > 0);
    /* for (size_t i=0; i<num_requests; i++) {
      for (size_t j=0; j<sequence_length; j++) {
        label_ptr[i * sequence_length + j] = std::rand() % num_labels;
      }
    } */
    uniform_int_distribution<int> int_dist{0, num_labels};
    auto gen_label = [&int_dist, &mersenne_engine]() {
      return int_dist(mersenne_engine);
    };
    std::generate(
        label_ptr, label_ptr + sequence_length * num_requests, gen_label);
  }
};

void DataGenerator::generate_arrival_times(void) {
  // set up a uniform number generator with range [0,1)
  random_device rnd;
  mt19937 gen(rnd());
  uniform_real_distribution<double> dist{0, 1.0};
  double cur_arrival = 0; // assume first request comes in at time 0

  for (size_t i = 0; i < num_requests; i++) {
    arrivals.push_back(cur_arrival);
    if (poisson_distr) {
      double u = dist(gen);
      double interval = -(1 / lambda) * log(1 - u) * 1000;
      cur_arrival += interval;
    } else {
      cur_arrival += (1000 / lambda);
    }
  }
  // cout << "Arrivals : [";
  // copy(arrivals.begin(), arrivals.end(), ostream_iterator<int>(cout, " "));
  // cout << "]" << endl;
};

void DataGenerator::start_timer(void) {
  arrivals_ptr = arrivals.begin();
  start_time = Clock::now();
  timer_started = true;
};

size_t DataGenerator::get_requests(size_t max_num_requests, std::vector<std::pair<size_t, std::vector<int> > >&prompts) {
  if (!timer_started) {
    std::cout << "Warning: tried to get number of requests before the timer "
                 "was started."
              << std::endl;
    return 0;
  }
  Clock::time_point cur_time = Clock::now();
  size_t ms_from_start =
      chrono::duration_cast<milliseconds>(cur_time - start_time).count();
  vector<double>::iterator new_arrivals_ptr =
      upper_bound(arrivals_ptr, arrivals.end(), ms_from_start);
  size_t received_requests = new_arrivals_ptr - arrivals_ptr;
  arrivals_ptr = new_arrivals_ptr;
  if (received_requests > 0) {
    std::cout << "received " << received_requests
              << " request(s) by arrival time +" << ms_from_start << "ms"
              << "\n";
  }

  for (size_t i = 0; i < received_requests; i++) {
    int length = std::rand() % 10 + 5;
    std::vector<int> prompt;
    for (int j = 0; j < length; j++)
      prompt.push_back(j + 1000);
    prompts.push_back(std::make_pair(global_unique_id++, prompt));
  }
  assert(prompts.size() == received_requests);
  return received_requests;
}

size_t DataGenerator::get_requests() {
  if (!timer_started) {
    std::cout << "Warning: tried to get number of requests before the timer "
                 "was started."
              << std::endl;
    return 0;
  }
  Clock::time_point cur_time = Clock::now();
  size_t ms_from_start =
      chrono::duration_cast<milliseconds>(cur_time - start_time).count();
  vector<double>::iterator new_arrivals_ptr =
      upper_bound(arrivals_ptr, arrivals.end(), ms_from_start);
  size_t received_requests = new_arrivals_ptr - arrivals_ptr;
  arrivals_ptr = new_arrivals_ptr;
  if (received_requests > 0) {
    std::cout << "received " << received_requests
              << " request(s) by arrival time +" << ms_from_start << "ms"
              << "\n";
  }

  return received_requests;
}
