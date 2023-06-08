import { getClasses, getNumbers } from 'ml-dataset-iris';

import SVM from '..';
import { leaveOneOut } from 'ml-cross-validation';

const labels = getClasses();
console.log(leaveOneOut(SVM, getNumbers(), labels.map(label => label == labels[0] ? 1 : -1), { kernel: 'rbf', C: 1, kernelOptions: { sigma: 0.2 } }));
