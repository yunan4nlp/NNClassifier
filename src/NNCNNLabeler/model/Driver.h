/*
* Driver.h
*
*  Created on: Mar 18, 2015
*      Author: mszhang
*/

#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include <iostream>
#include "ComputionGraph.h"


//A native neural network classfier using only word embeddings

class Driver{
public:
	Driver(int memsize) :_aligned_mem(memsize){
		_pcg = NULL;
	}

	~Driver() {
		if (_pcg != NULL)
			delete _pcg;
		_pcg = NULL;
	}

public:
	ComputionGraph *_pcg;  // build neural graphs
	ModelParams _modelparams;  // model parameters
	HyperParams _hyperparams;

	Metric _eval;
	CheckGrad _checkgrad;
	ModelUpdate _ada;  // model update
	AlignedMemoryPool _aligned_mem;


public:
	//embeddings are initialized before this separately.
	inline void initial() {
		if (!_hyperparams.bValid()){
			std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
			return;
		}
		if (!_modelparams.initial(_hyperparams, &_aligned_mem)){
			std::cout << "model parameter initialization Error, Please check!" << std::endl;
			return;
		}
		_modelparams.exportModelParams(_ada);
		_modelparams.exportCheckGradParams(_checkgrad);

		_hyperparams.print();

		_pcg = new ComputionGraph();
		_pcg->createNodes(ComputionGraph::max_sentence_length);
		_pcg->initial(_modelparams, _hyperparams, &_aligned_mem);

		setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
	}

	inline void TestInitial() {
		if (!_hyperparams.bValid()){
			std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
			return;
		}
		if (!_modelparams.TestInitial(_hyperparams, &_aligned_mem)){
			std::cout << "model parameter initialization Error, Please check!" << std::endl;
			return;
		}
		_modelparams.exportModelParams(_ada);
		_modelparams.exportCheckGradParams(_checkgrad);

		_hyperparams.print();

		_pcg = new ComputionGraph();
		_pcg->createNodes(ComputionGraph::max_sentence_length);
		_pcg->initial(_modelparams, _hyperparams, &_aligned_mem);

		setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
	}

	inline dtype train(const vector<Example>& examples, int iter) {
		_eval.reset();

		int example_num = examples.size();
		dtype cost = 0.0;

		for (int count = 0; count < example_num; count++) {
			const Example& example = examples[count];

			//forward
			_pcg->forward(example.m_feature, true);

			//loss function
			//for (int idx = 0; idx < seq_size; idx++) {
			//cost += _loss.loss(&(_pcg->_output[idx]), example.m_labels[idx], _eval, example_num);				
			//}
			cost += _modelparams.loss.loss(&_pcg->_output, example.m_label, _eval, example_num);

			// backward, which exists only for training 
			_pcg->backward();
		}

		if (_eval.getAccuracy() < 0) {
			std::cout << "strange" << std::endl;
		}

		return cost;
	}

	inline void predict(const Feature& feature, int& result) {
		_pcg->forward(feature);
		//results.resize(seq_size);
		//for (int idx = 0; idx < seq_size; idx++) {
		//	_loss.predict( &(_pcg->_output[idx]), results[idx]);
		//}
		_modelparams.loss.predict(&_pcg->_output, result);
	}

	inline dtype cost(const Example& example){
		_pcg->forward(example.m_feature); //forward here


		dtype cost = 0.0;
		//loss function
		//for (int idx = 0; idx < seq_size; idx++) {
		//	cost += _loss.cost(&(_pcg->_output[idx]), example.m_labels[idx], 1);
		//}
		cost += _modelparams.loss.cost(&_pcg->_output, example.m_label, 1);

		return cost;
	}


	void updateModel() {
		//_ada.update();
		_ada.update(5.0);
	}

	void checkgrad(const vector<Example>& examples, int iter){
		ostringstream out;
		out << "Iteration: " << iter;
		_checkgrad.check(this, examples, out.str());
	}

	void writeModel();

	void loadModel();



private:
	inline void resetEval() {
		_eval.reset();
	}


	inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps){
		_ada._alpha = adaAlpha;
		_ada._eps = adaEps;
		_ada._reg = nnRegular;
	}

};

#endif /* SRC_Driver_H_ */
