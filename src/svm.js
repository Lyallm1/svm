import Kernel from 'ml-kernel';
import { array as stat } from 'ml-stat';

export default class SVM {
    constructor(options) {
        this.options = Object.assign({}, { C: 1, tol: 1e-4, maxPasses: 10, maxIterations: 10000, kernel: 'linear', alphaTol: 1e-6, random: Math.random, whitening: true }, options);
        this.kernel = new Kernel(this.options.kernel, this.options.kernelOptions);
        this.b = 0;
    }

    static load(model) {
        this._loaded = true;
        this._trained = false;
        const svm = new SVM(model.options);
        if (model.options.kernel === 'linear') {
            svm.W = model.W.slice();
            svm.D = svm.W.length;
        } else {
            svm.X = model.X.slice();
            svm.Y = model.Y.slice();
            svm.alphas = model.alphas.slice();
            svm.N = svm.X.length;
            svm.D = svm.X[0].length;
        }
        svm.minMax = model.minMax;
        svm.b = model.b;
        svm._loaded = true;
        svm._trained = false;
        return svm;
    }
    
    train(features, labels) {
        if (features.length !== labels.length) throw new Error('Features and labels should have the same length');
        if (features.length < 2) throw new Error('Cannot train with less than 2 observations');
        this._trained = false;
        this._loaded = false;
        this.N = labels.length;
        this.D = features[0].length;
        if (this.options.whitening) {
            this.X = new Array(this.N).fill(new Array(this.D));
            this.minMax = new Array(this.D);
            for (let j = 0; j < this.D; j++) {
                this.minMax[j] = stat.minMax(Array.from({ length: this.N }, i => features[i][j]));
                for (i = 0; i < this.N; i++) this.X[i][j] = (features[i][j] - this.minMax[j].min) / (this.minMax[j].max - this.minMax[j].min);
            }
        } else this.X = features;
        this.Y = labels;
        this.b = 0;
        this.W = undefined;
        const kernel = this.kernel.compute(this.X), m = labels.length, alpha = Array.from(labels, () => 0)
        this.alphas = alpha;
        let b1 = 0, b2 = 0, iter = 0, passes = 0, Ei = 0, Ej = 0, ai = 0, aj = 0, L = 0, H = 0, eta = 0;
        while (passes < this.options.maxPasses && iter < this.options.maxIterations) {
            let numChange = 0;
            for (i = 0; i < m; i++) {
                Ei = this._marginOnePrecomputed(i, kernel) - labels[i];
                if (labels[i] * Ei < -this.options.tol && alpha[i] < this.options.C || labels[i] * Ei > this.options.tol && alpha[i] > 0) {
                    j = i;
                    while (j === i) j = Math.floor(this.options.random() * m);
                    Ej = this._marginOnePrecomputed(j, kernel) - labels[j];
                    ai = alpha[i];
                    aj = alpha[j];
                    L = Math.max(0, labels[i] === labels[j] ? ai + aj - this.options.C : aj - ai);
                    H = Math.min(this.options.C, labels[i] === labels[j] ? ai + aj : this.options.C + aj + ai);
                    if (Math.abs(L - H) < 1e-4) continue;
                    eta = 2 * kernel[i][j] - (kernel[i][i] + kernel[j][j]);
                    if (eta >= 0) continue;
                    let newaj = alpha[j] - labels[j] * (Ei - Ej) / eta;
                    if (newaj > H) newaj = H;
                    else if (newaj < L) newaj = L;
                    if (Math.abs(aj - newaj) < 10e-4) continue;
                    alpha[j] = newaj;
                    alpha[i] += labels[i] * labels[j] * (aj - newaj);
                    b1 = this.b - Ei - labels[i] * (alpha[i] - ai) * kernel[i][i] - labels[j] * (alpha[j] - aj) * kernel[i][j];
                    b2 = this.b - Ej - labels[i] * (alpha[i] - ai) * kernel[i][j] - labels[j] * (alpha[j] - aj) * kernel[j][j];
                    this.b = (b1 + b2) / 2;
                    if (alpha[i] < this.options.C && alpha[i] > 0) this.b = b1;
                    if (alpha[j] < this.options.C && alpha[j] > 0) this.b = b2;
                    numChange++;
                }
            }
            iter++;
            if (numChange === 0) passes++;
            else passes = 0;
        }
        if (iter === this.options.maxIterations) throw new Error('max iterations reached');
        this.iterations = iter;
        if (this.options.kernel === 'linear') {
            this.W = new Array(this.D).fill(0);
            for (let r = 0; r < this.D; r++) for (let w = 0; w < m; w++) this.W[r] += labels[w] * alpha[w] * this.X[w][r];
        }
        const nX = [], nY = [], nAlphas = [];
        this._supportVectorIdx = [];
        for (i = 0; i < this.N; i++) if (this.alphas[i] > this.options.alphaTol) {
            nX.push(this.X[i]);
            nY.push(labels[i]);
            nAlphas.push(this.alphas[i]);
            this._supportVectorIdx.push(i);
        }
        this.X = nX;
        this.Y = nY;
        this.N = nX.length;
        this.alphas = nAlphas;
        this._trained = true;
    }
    
    predictOne(p) {
        return this.marginOne(p) > 0 ? 1 : -1;
    }
    
    predict(features) {
        if (!(this._trained || this._loaded)) throw new Error('Cannot predict, you need to train the SVM first');
        return Array.isArray(features) && Array.isArray(features[0]) ? features.map(this.predictOne.bind(this)) : this.predictOne(features);
    }
    
    marginOne(features, noWhitening) {
        if (this.options.whitening && !noWhitening) features = this._applyWhitening(features);
        let ans = this.b;
        if (this.options.kernel === 'linear' && this.W) this.W.forEach((val, i) => ans += val * features[i]);
        else for (let i = 0; i < this.N; i++) ans += this.alphas[i] * this.Y[i] * this.kernel.compute([features], [this.X[i]])[0][0];
        return ans;
    }
    
    _marginOnePrecomputed(index, kernel) {
        let ans = this.b;
        for (let i = 0; i < this.N; i++) ans += this.alphas[i] * this.Y[i] * kernel[index][i];
        return ans;
    }
    
    margin(features) {
        Array.isArray(features) ? features.map(this.marginOne.bind(this)) : this.marginOne(features);
    }
    
    supportVectors() {
        if (!(this._trained || this._loaded)) throw new Error('Cannot get support vectors, you need to train the SVM first');
        if (this._loaded && this.options.kernel === 'linear') throw new Error('Cannot get support vectors from saved linear model, you need to train the SVM to have them');
        return this._supportVectorIdx;
    }
    
    toJSON() {
        if (!(this._trained || this._loaded)) throw new Error('Cannot export, you need to train the SVM first');
        const model = { options: Object.assign({}, this.options), b: this.b, minMax: this.minMax };
        if (model.options.kernel === 'linear') model.W = this.W.slice();
        else {
            model.X = this.X.slice();
            model.Y = this.Y.slice();
            model.alphas = this.alphas.slice();
        }
        return model;
    }
    
    _applyWhitening(features) {
        if (!this.minMax) throw new Error('Could not apply whitening');
        return Array.from(features, (feature, j) => (feature - this.minMax[j].min) / (this.minMax[j].max - this.minMax[j].min));
    }
}
