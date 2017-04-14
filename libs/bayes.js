'use strict';

module.exports = function () {
    let app = this;

	// logger
    let assert = (condition, message)=> {
        if (!condition) {
            message = message || "Assertion failed";
            if (typeof Error !== "undefined") {
                throw new Error(message);
            }
            throw message;
        }
    };

    let _trained = null;

	// set previous trained model
    app.setModel = (trained)=> {
        assert(trained.constructor == Object, `dataset undefined`);
        _trained = trained;
    };

	// get trained model
    app.getModel = ()=> _trained;

	// training function
    app.train = (dataset, labels)=> {
        assert(dataset, `dataset undefined`);
        assert(labels, `labels undefined`);

		// check dataset structure
        if (Array.isArray(dataset) === false) dataset = [dataset];
        else if (typeof dataset[0] != 'object') dataset = [dataset];

        if (Array.isArray(labels) === false) labels = [labels];

        assert(dataset.length === labels.length, `mismatched array length`);

        if (!_trained) _trained = {};
        if (!_trained.dic) _trained.dic = {};
        if (!_trained.label) _trained.label = {};
        
		let _dic = _trained.dic; // dictionary map
        let _label = _trained.label; // labels data

		// dataset iteration
        for (let i = 0; i < dataset.length; i++) {
            let data = dataset[i]; // single data row
            let label = labels[i]; // data's label

            let sum = 0;

			// features in data row
            for (let key in data) {
				// if not contained in dictionary, create map data as zero
                if (!_dic[key]) _dic[key] = {};
                if (!_dic[key][label]) _dic[key][label] = 0;
				
				// add features value in dictionary
                _dic[key][label] += data[key] * 1;
                
				// add features value
				sum += data[key] * 1;
            }

			// label ratio 
            if (!_label[label]) _label[label] = 0;
            _label[label] += sum;
        }

        return _trained;
    };

	// predict some data, core calculation function in here.
    app.test = (dataset, options)=> {
        if (!options) options = {};

        assert(dataset, `dataset undefined`);
        if (Array.isArray(dataset) === false) dataset = [dataset];
        else if (typeof dataset[0] != 'object') dataset = [dataset];

		// load trained model
        let _dic = _trained.dic;
        let _label = _trained.label;

        let result = [];

		// test dataset iteration
        for (let i = 0; i < dataset.length; i++) {
            let prob = {};
            let data = dataset[i];
            
			// calculate prob for each label
			for (let label in _label) {
                if (typeof prob[label] === 'undefined') prob[label] = 0;
                for (let key in data) {
                    let fc = _dic[key] ? _dic[key][label] ? _dic[key][label] * data[key] : 0 : 0;
                    fc += 1;
                    prob[label] += Math.log(fc / _label[label]);
                }
            }

            let max = null;
            let answer = null;

			// select max scored label
            for (let label in prob) {
                if (!max) {
                    max = prob[label];
                    answer = label;
                }

                if (prob[label] > max) {
                    max = prob[label];
                    answer = label;
                }
            }

            if (options.score) result.push({answer: answer, score: prob});
            else result.push(answer);
        }

        if (dataset.length == 1) return result[0];
        else return result;
    };

    return app;
};
