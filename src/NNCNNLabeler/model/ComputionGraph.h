#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct ComputionGraph : Graph{
public:
	const static int max_sentence_length = 2048;

public:
	// node instances
	vector<LookupNode> _word_inputs;
	WindowBuilder _word_window;
	vector<UniNode> _hidden;

	AvgPoolNode _avg_pooling;
	MaxPoolNode _max_pooling;
	MinPoolNode _min_pooling;

	ConcatNode _concat;

	LinearNode _neural_output;
	SparseNode _sparse_output;
	Node _sparse_bucket;
	PAddNode _output;
public:
	ComputionGraph() : Graph(){
	}

	~ComputionGraph(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length){
		_word_inputs.resize(sent_length);
		_word_window.resize(sent_length);
		_hidden.resize(sent_length);
		_avg_pooling.setParam(sent_length);
		_max_pooling.setParam(sent_length);
		_min_pooling.setParam(sent_length);
	}

	inline void clear(){
		Graph::clear();
		_word_inputs.clear();
		_word_window.clear();
		_hidden.clear();
	}

public:
	inline void initial(ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem = NULL){
		for (int idx = 0; idx < _word_inputs.size(); idx++) {
			_word_inputs[idx].setParam(&model.words);
			_word_inputs[idx].init(opts.wordDim, opts.dropProb, mem);
			_hidden[idx].setParam(&model.hidden_linear);
			_hidden[idx].init(opts.hiddenSize, opts.dropProb, mem);
		}
		_word_window.init(opts.wordDim, opts.wordContext, mem);
		_avg_pooling.init(opts.hiddenSize, -1, mem);
		_max_pooling.init(opts.hiddenSize, -1, mem);
		_min_pooling.init(opts.hiddenSize, -1, mem);
		_concat.init(opts.hiddenSize * 3, -1, mem);
		_neural_output.setParam(&model.olayer_linear);
		_neural_output.init(opts.labelSize, -1, mem);

		_sparse_output.setParam(&model.sparse_params);
		_sparse_output.init(opts.labelSize, -1, mem);
		_sparse_bucket.init(opts.labelSize, -1, mem);
		_sparse_bucket.set_bucket();

		_output.init(opts.labelSize, -1, mem);
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const Feature& feature, bool bTrain = false){
		//first step: clear value
		clearValue(bTrain); // compute is a must step for train, predict and cost computation

		// second step: build graph
		//forward
		int words_num = feature.m_words.size();
		if (words_num > max_sentence_length)
			words_num = max_sentence_length;
		for (int i = 0; i < words_num; i++) {
			_word_inputs[i].forward(this, feature.m_words[i]);
		}
		_word_window.forward(this, getPNodes(_word_inputs, words_num));

		for (int i = 0; i < words_num; i++) {
			_hidden[i].forward(this, &_word_window._outputs[i]);
		}
		_avg_pooling.forward(this, getPNodes(_hidden, words_num));
		_max_pooling.forward(this, getPNodes(_hidden, words_num));
		_min_pooling.forward(this, getPNodes(_hidden, words_num));
		_concat.forward(this, &_avg_pooling, &_max_pooling, &_min_pooling);
		_neural_output.forward(this, &_concat);
		
		if (feature.m_sparse_feats.size() > 0)
		{
			_sparse_output.forward(this, feature.m_sparse_feats);
			_output.forward(this, &_neural_output, &_sparse_output);
		}
		else
			_output.forward(this, &_neural_output, &_sparse_bucket);
	}
};

#endif /* SRC_ComputionGraph_H_ */