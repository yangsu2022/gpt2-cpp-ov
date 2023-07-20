#include "tqdm.h"
#include "config.hpp"
#include "GPT2Tokenizer.hpp"
#include "cxxopts.hpp"

#include <chrono>
#include <limits>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>

#include <openvino/openvino.hpp>
#include "openvino/core/preprocess/preprocess_steps.hpp"
#include "openvino/opsets/opset8.hpp"

static constexpr size_t BATCH_SIZE = 1;

std::vector<double> normalize_probabilities(const std::vector<double>& probabilities) {
    std::vector<double> norm_probabilities = probabilities;
    double sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    
    if (sum > 0.0) {
        for (double& probability : norm_probabilities) {
            probability /= sum;
        }
    }
    return norm_probabilities;
}


int get_rand_index(const std::vector<double>& probabilities) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());

    int index = dist(gen);
    return index;
}


template <typename T>
T unwrap(std::optional<T>&& value, const std::string& error_msg) {
    if (value.has_value()) {
        return value.value();
    }
    else {
        throw std::runtime_error(error_msg);
    }
} 


template <typename T>
struct view {
    typename std::vector<T>::iterator _start;
    typename std::vector<T>::iterator _end;

    auto begin() const {
        return _start;
    }
    auto end() const {
        return _end;
    }
};


auto softmax = [](view<float> vec) { 
    std::transform(vec.begin(), vec.end(), vec.begin(), [](const float& el){ return std::exp(el); });
    const float sum = std::accumulate(vec.begin(), vec.end(), 0.f);
    std::transform(vec.begin(), vec.end(), vec.begin(), [sum](const float& el){ return el/sum; });
};


size_t ov_next_token_prediction(ov::CompiledModel compiled_model, std::vector<int64_t>& token_ids, const size_t vocab_size) {
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    auto input_port = compiled_model.input();
    ov::Shape input_shape = {1, 1, token_ids.size()};
    ov::Tensor input_tensor(input_port.get_element_type(), input_shape, token_ids.data()); // get_shape Aborted
    infer_request.set_tensor(input_port, input_tensor);
    infer_request.start_async();
    infer_request.wait();
    
    auto output_port = compiled_model.outputs()[0];
    ov::Tensor output_tensor = infer_request.get_tensor(output_port);
   
    std::vector<float> prediction_scores;
    auto out_data = output_tensor.data<float>();  

    for (size_t i = 0; i < BATCH_SIZE * token_ids.size() * vocab_size; i++) {
        prediction_scores.push_back(out_data[i]);
    }

    auto new_word_prediction = view<float>{prediction_scores.begin() + (token_ids.size()-1) * vocab_size,  prediction_scores.end()};
    std::vector<std::pair<float, int>> indexedNumbers;
    for (int i = 0; i < static_cast<int>(vocab_size); ++i) {
        indexedNumbers.emplace_back(*(new_word_prediction.begin() + i), i);
    }
    int topk = 20;
    
    std::nth_element(indexedNumbers.begin(), 
        indexedNumbers.begin() + topk, 
        indexedNumbers.end(),
        [](const auto& a, const auto& b) {return a.first > b.first;});
    
    std::vector<int> indices;
    std::vector<float> topk_vals;
    std::vector<float> scaled_topk_vals;
    float inf = std::numeric_limits<double>::infinity();
 
    for (int i = 0; i < static_cast<int>(indexedNumbers.size()); ++i) {
        if (indexedNumbers[i].first > indexedNumbers[topk].first) {
            topk_vals.push_back(indexedNumbers[i].first);
            indices.push_back(indexedNumbers[i].second);
        } 
    }
    const auto max_vals = std::max_element(topk_vals.begin(), topk_vals.end());

    for (float vals : topk_vals) {scaled_topk_vals.push_back(vals - *max_vals);}
    // std::cout << std::endl;
    auto new_vals = view<float>{scaled_topk_vals.begin(), scaled_topk_vals.end()};
    
    int position = std::distance(new_vals.begin(), max_vals);

    double maxInput = *std::max_element(new_vals.begin(), new_vals.end());

    softmax(new_vals);

    std::vector<double> probabilities;
    
    for (size_t i = 0; i < topk; i++) {
        probabilities.push_back(*(new_vals.begin()+i));
    }

    std::vector<double> norm_probabilities = normalize_probabilities(probabilities);
    int rand_index = get_rand_index(norm_probabilities);
    // std::cout << "rand_index: " << rand_index << std::endl;

    const auto new_max_vals = std::max_element(new_vals.begin(), new_vals.end());
    int new_position = std::distance(new_vals.begin(), new_max_vals);

    return indices[rand_index];

}


size_t ov_next_token_prediction_ppp(ov::CompiledModel compiled_model, std::vector<int64_t>& token_ids, const size_t vocab_size) {
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // Get input port for model with one input
    auto input_port = compiled_model.input();
    // Create tensor from external memory
    ov::Shape input_shape = {1, 1, token_ids.size()};
    ov::Tensor input_tensor(input_port.get_element_type(), input_shape, token_ids.data()); // get_shape Aborted
    infer_request.set_tensor(input_port, input_tensor);
    infer_request.start_async();
    infer_request.wait();
    // infer_request.infer();
    auto output_port_0 = compiled_model.outputs()[0];
    ov::Tensor output_tensor = infer_request.get_tensor(output_port_0);
    ov::Tensor output_tensor_0 = infer_request.get_output_tensor(0);
    
    std::vector<float> topk_values;
    std::vector<float> topk_indices;
    
    auto out_data_topk = output_tensor.data<float>(); 
    

    int topk = 20;
    for (size_t i = 0; i < topk; i++) {
        topk_values.push_back(out_data_topk[i]);
    }
    for (size_t i = topk; i < 2 * topk; i++) {
        topk_indices.push_back(out_data_topk[i]);
    }

    std::vector<double> probabilities;
    for (size_t i = 0; i < topk; i++) {
        probabilities.push_back(*(topk_values.begin()+i));
    }

    const auto new_max_vals = std::max_element(topk_values.begin(), topk_values.end());
    int new_position = std::distance(topk_values.begin(), new_max_vals);
    
    std::vector<double> normalized_probabilities = normalize_probabilities(probabilities);
    int rand_index = get_rand_index(normalized_probabilities);
    // std::cout << "rand_index: " << rand_index << std::endl;
    
    return topk_indices[rand_index];
}


int main(int argc, char *argv[]) {

    cxxopts::Options options("GPT2", "GPT2 implementation in C++ using Ort");

    options.add_options()
        ("t,text", "Initial text for GPT2", cxxopts::value<std::string>())
        ("n,number", "Number of new words to generate from initial text", cxxopts::value<size_t>()->default_value("1"))
        ("p,ppp", "Optimize with PPP for topK", cxxopts::value<bool>())
        ("h,help", "Print usage")
    ;
    cxxopts::ParseResult result;
    
    try {
        result = options.parse(argc, argv);
    } catch (const cxxopts::OptionException& e) {
        std::cout << e.what() << "\n\n";
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (result.count("text") == 0) {
        std::cout << "Expected text input!\n\n";
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (result.count("ppp") != 0) {
        std::cout << "Optimizing model with PrePostProcessor\n";
    }

    const std::string text = result["text"].as<std::string>();
    const size_t generate = result["number"].as<size_t>();
    const bool set_ppp = result["ppp"].as<bool>();

    auto tokenizer = unwrap(GPT2Tokenizer::load(vocab_file, merges_file), "Error initialising GPT2 tokenizer\n");
    auto token_ids = tokenizer.encode(text);

    std::cout << "using OpenVINO runtime with topK sampling(k=20)\n\n";

    tqdm bar;
    double sum_duration = 0.0;
    // this ONNX model is exported with ORT 1.10 
    // https://github.com/onnx/models/blob/main/text/machine_comprehension/gpt-2/dependencies/GPT2-export.py
    // std::string model_path = "/home/fiona/workspaces/ys/gpt2-cpp/data/gpt2-lm-head-10.onnx"; 
    std::string model_path = "/home/fiona/workspaces/ys/gpt2-cpp/data/IR_FP32/gpt2-lm-head-10.xml"; 

    ov::Core core;
    core.set_property(ov::cache_dir("./cache_dir")); 
    std::shared_ptr<ov::Model> model = core.read_model(model_path.c_str());
    
    if (set_ppp) {
        ov::preprocess::PrePostProcessor ppp(model);    
        ppp.output("output1").postprocess()
        .custom([](const ov::Output<ov::Node>& node) {
            // Custom nodes can be inserted as Post-processing steps
            auto input_shape = std::make_shared<ov::opset8::ShapeOf>(node, ov::element::i64);
            // use Slice node to get the token size
            auto token_size_start = ov::opset8::Constant::create(ov::element::i64, ov::Shape{1}, {2}); 
            auto token_size_end = ov::opset8::Constant::create(ov::element::i64, ov::Shape{1}, {3}); 
            auto token_size_step = ov::opset8::Constant::create(ov::element::i64, ov::Shape{1}, {1}); 
            auto token_size_axis = ov::opset8::Constant::create(ov::element::i64, ov::Shape{1}, {0}); 
            auto token_size = std::make_shared<ov::opset8::Slice>(
                input_shape, token_size_start, token_size_end, token_size_step, token_size_axis); 
            
            // use Slice node to get the new token without input token 
            auto constant_start = ov::opset8::Constant::create(ov::element::i64, ov::Shape{1}, {-1}); 
            auto new_token_start = std::make_shared<ov::opset8::Add>(token_size, constant_start);
            auto new_token_step = ov::opset8::Constant::create(ov::element::i64, ov::Shape{1}, {1}); 
            auto new_token_axis = ov::opset8::Constant::create(ov::element::i64, ov::Shape{1}, {2}); 
            auto new_token = std::make_shared<ov::opset8::Slice>(
                node, new_token_start, token_size, new_token_step, new_token_axis);

            // use TopK(k=20) node to optimize the postprocessing
            auto topk_constant_k = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {20}); 
            auto topk_node = std::make_shared<ov::opset8::TopK>(
                new_token, topk_constant_k,3,"max","none", ov::element::i32); 
            auto topk_node_value = topk_node->outputs()[0];
            auto topk_node_index = topk_node->outputs()[1];
            auto new_topk_node_index = std::make_shared<ov::opset8::Convert>(
                topk_node_index, ov::element::f32);
            
            // use Softmax without norm
            auto softmax_node = std::make_shared<ov::opset8::Softmax>(topk_node_value, 3);

            // use Concat node to combine the Softmax output and TopK indices
            auto new_concat_node = std::make_shared<ov::opset8::Concat>(
                ov::OutputVector{softmax_node, new_topk_node_index}, 3);
            
            return new_concat_node;

        });
        model = ppp.build(); 
    }

    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");

    for (size_t i = 0; i < generate; ++i) {
        bar.progress(i, generate);

        auto start = std::chrono::high_resolution_clock::now();
        if (set_ppp) { 
            token_ids.push_back(
                ov_next_token_prediction_ppp(
                    compiled_model, token_ids, tokenizer.vocab_size()));
        } else {
            token_ids.push_back(
                ov_next_token_prediction(
                    compiled_model, token_ids, tokenizer.vocab_size()));
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
        sum_duration += duration;
        }
       
    bar.finish();
    double avg_duration = sum_duration / generate;
    std::cout << "MEAN time of single token prediction: " << avg_duration << " ms" << std::endl;
    std::cout << "OV Prediction: \"" << tokenizer.decode(token_ids) << '\"' << '\n';
}
