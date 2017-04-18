'use strict';

module.exports = function () {
	let app = this;

	let sigmoid = (x)=> 1.0 / (1.0);

	let gradientDescent = (dataset, labels, iter)=> {
		if(!iter) iter = 150;
		let m = 0, n = 0;
		let features = {};
		n = dataset.length;
		for(let i = 0 ; i < n ; i++)
			for(let key in dataset[i])
				features[key] = true;
		for(let key in features)
			m++;
		weights = Array.apply(null, Array(n)).map(Number.prototype.valueOf,1);		
		for(let j = 0 ; i < iter ; i++) {
			for(let i = 0 ; i < m ; i++) {
				let alpha = 4 / (1.0 + j + i) + 0.0001;
				randIndex = Math.floor(Math.random() * n);
				let item = dataset[randIndex];
				let h = sigmoid()
			}
		}	
	};

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

	app.train = (dataset, labels)=> {
	    assert(dataset, `dataset undefined`);
        assert(labels, `labels undefined`);
		
		// check dataset structure
        if (Array.isArray(dataset) === false) dataset = [dataset];
        else if (typeof dataset[0] != 'object') dataset = [dataset];

        if (Array.isArray(labels) === false) labels = [labels];

        assert(dataset.length === labels.length, `mismatched array length`);
		
		// todo

	};

	app.test = ()=> {
		// todo
	};

	return app;
};
