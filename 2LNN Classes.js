class ResNN {
    constructor(){

        //CONTROL VARIABLES
        //Input vector
        this.input = [.9,.7,.1,0,0,.1,0,.36];

        this.inputSize = 50;

        this.rcOn = true;

        //Vector the AI is training to match
        this.targetVector = [.5,0,0,(0-.5),0,0,.3,0];

        //Hyperparameters
        this.learningFactor = .1;
        this.percentUpdated = .008;
        this.bias = .0001;
        this.reservoirPercentage = .1;
        this.reservoirGearRatio = 4;

        //Loop iterations
        this.runs = 1000;
        ////////////////////////

        //MAIN VARIABLES
        this.inputVector = [];

        this.errorVectorL1 = [];
        this.errorVectorL2 = [];

        this.errorGradientL2 = [];

        this.outputVector1 = [];
        this.outputVector2 = [];

        this.outputVectorA1 = [];
        this.outputVectorA2 = [];

        this.dOutput1 = [];

        this.transformedVector = [];
        this.transformedVectorTemp = [];
    }
//END CONSTRUCTOR
//END PADDING

//MATRIX GENERATION UTIL FUNCTIONS
//Matrix generator that randomizes all cells
create0Matrix(diameter){
    var matrix = [];

    for (let i = 0; i < diameter; i++) {
        matrix[i] = [];
        for(let k = 0; k < diameter; k++) {
            matrix[i][k] = Math.random() - .5;
        }
      }
    return matrix;
}

//Matrix generator that zeroes all cells
createZMatrix(diameter){
    var matrix = [];

    for (let i = 0; i < diameter; i++) {
        matrix[i] = [];
        for(let k = 0; k < diameter; k++) {
            matrix[i][k] = 0;
        }
      }
    return matrix;
}
//END MATRIX GENERATION UTIL FUNCTIONS

//MISC FUNCTIONS
dActivation(x){
    return 4/(Math.pow((Math.exp((0-1)*x)+Math.exp(x)),2));
}
//END MISC FUNCTIONS

createResMatrix(diameter){
    var matrix = [];

    for (let i = 0; i < diameter; i++) {
        matrix[i] = [];
        for(let j = 0; j < diameter; j++) {
            if(Math.random() < reservoirPercentage){
                matrix[i][j] = Math.random() - .5;
            }else{
                matrix[i][j] = 0;
            }
        }
      }
    return matrix;
}

//BEGIN FORWARD PASS
//Matrix Multiplies Input Vector by Weight Matrix and Applies Standard Tanh Activation Function
matrixMultiply1(inputV, weightM, diam){
    var outputVector = [];
    for(let i = 0; i < diam; i++){
        var sum = 0;
        for(let k = 0; k < diam; k++){
            sum = sum + (inputV[k] * weightM[i][k]);
        }
        outputVector1[i] = sum;
        outputVectorA1[i] = Math.tanh(sum + bias);
        outputVector[i] = Math.tanh(sum + bias);
    }
    return outputVector;
}

matrixMultiply2(inputV, weightM, diam){
    var outputVector = [];
    for(let i = 0; i < diam; i++){
        var sum = 0;
        for(let k = 0; k < diam; k++){
            sum = sum + (inputV[k] * weightM[i][k]);
        }
        outputVector2[i] = sum;
        outputVectorA2[i] = Math.tanh(sum + bias);
        outputVector[i] = Math.tanh(sum + bias);
    }
    return outputVector;
}


//END FORWARD PASS

//ACKWARD PASS UTIL FUNCTIONS


//RESERVOIR

//Reservoir computer preprocesses temporally correlated inputs
createResMatrix(diameter){
    var matrix = [];

    for (let i = 0; i < diameter; i++) {
        matrix[i] = [];
        for(let k = 0; k < diameter; k++) {
            if(Math.random() < reservoirPercentage){
                matrix[i][k] = Math.random() - .5;
            }else{
                matrix[i][k] = 0;
            }
        }
      }
    return matrix;
}

updateReservoirProjection(inputV, weightM, diam, zeroIndex, initialInputProjection){
    var outputVector = [];
    for(let i = 0; i < diam; i++){
        if(i >= zeroIndex){
            inputV[i] = initialInputProjection[i] + inputV[i];
        }
    }
    for(let i = 0; i < diam; i++){
        var sum = 0;
        for(let j = 0; j < diam; j++){
            sum = sum + (inputV[j] * weightM[i][j]);
        }
        outputVector[i] = Math.tanh(sum + bias);
    }
    initialInputProjection = outputVector;
    return outputVector;
}

normalizeInputs(projectedInputs){
    var sum = 0;
    for(let i = 0; i < projectedInputs.length; i++){
        sum = sum + projectedInputs[i];
    }
    for(let i = 0; i < projectedInputs.length; i++){
        projectedInputs[i] = projectedInputs[i] / sum;
    }
    return projectedInputs;
}                                                                                                            
//RESERVOIR

backPropagateO(arg1, arg2, arg3, arg4){
    let dError = [];
    let bpWeights = [];
    let outV = []
    let dOutput = [];
    bpWeights = arg1;
    dError = arg2;
    outV = arg3;
    dOutput = arg4;

    
    for(let i = 0; i < inputSize; i++){
        for(let j = 0; j < inputSize; j++){
            gradientMatrix2[i][j] = (learningFactor*outputVector1[j]*dOutput[i][j]*dError[i][j]);
        }
    }

    for(let i = 0; i < inputSize; i++){
        for(let j = 0; j < inputSize; j++){
            bpWeights[i][j] = bpWeights[i][j] + gradientMatrix2[i][j];
        }
    }

    return bpWeights;
}



backPropagateH1(arg1, arg3, arg4, arg5){
    let dErrorMost = createZMatrix(inputSize);
    let bpWeights = [];
    let outV = []
    let dOutput = [];
    let outputA = [];
    let E = 0;
    bpWeights = arg1;
    outV = arg3;
    dOutput = arg4;
    outputA = arg5;


    //Calculate dErrorMost//
    for(let i = 0; i < inputSize; i++){
        E = 0;
        for(let j = 0; j < inputSize; j++){
                E = E + weightMatrix2Transpose[i][j]*dErrorVectr2[i][j]*dOutput2[i][j];
            dErrorMost[i][j] = E;
        }
    }

    for(let i = 0; i < inputSize; i++){
        for(let j = 0; j < inputSize; j++){
            gradientMatrix1[i][j] = (learningFactor*normalizedProjectedInput[i]*dOutput[j]*dErrorMost[i][j]);
        }
    }

    for(let i = 0; i < inputSize; i++){
        for(let j = 0; j < inputSize; j++){
            bpWeights[i][j] = bpWeights[i][j] + gradientMatrix1[i][j];
        }
    }

    return bpWeights;
}



//END BACKWARD PASS UTIL FUNCTIONS
train(){
this.gradientMatrix2 = createZMatrix(inputSize);
this.gradientMatrix1 = createZMatrix(inputSize);
//END MAIN VARIABLES

inputVector = input;

//PAD INPUT AND TARGET VECTORS
for(let i = input.length; i < inputSize; i++){
    inputVector[i] = 0;
}

for(let i = targetVector.length; i < inputSize; i++){
    targetVector[i] = 0;
}

this.dErrorVectr2 = createZMatrix(inputSize);
this.dOutput2 = createZMatrix(inputSize);

//New randomized square weight matrix
let weightMatrix1 = create0Matrix(inputSize);
let weightMatrix2 = create0Matrix(inputSize);
let weightMatrix2Transpose = create0Matrix(inputSize);

let reservoirMatrix = createResMatrix(inputSize);
let projectedInputVector = [];
for(let i = 0; i < inputSize; i++){
    projectedInputVector[i] = 0;
}
let normalizedProjectedInput = [];


for(let r = 0; r < runs; r++){
    var errorInitial = 0;
    var tempError = 0;
    var finalError = 0;

    if(rcOn == true){
        projectedInputVector = updateReservoirProjection(inputVector, reservoirMatrix, inputSize, input.length, projectedInputVector);
        normalizedProjectedInput = normalizeInputs(projectedInputVector);
        reservoirGearRatio = 1;
        //console.log(normalizedProjectedInput);
    } else{
        normalizedProjectedInput = inputVector;
    }

    for(let a = 0; a < reservoirGearRatio; a++){

        transformedVector = matrixMultiply2(matrixMultiply1(normalizedProjectedInput, weightMatrix1, inputVector.length),weightMatrix2,inputVector.length);
        //console.log(transformedVector);

        for(let i = 0; i < inputSize; i++){
            dOutput1[i] = dActivation(outputVector1[i]);
        }

    
        for(let i = 0; i < inputSize; i++){
            for(let j = 0; j < inputSize; j++){
                dOutput2[i][j] = dActivation(outputVector2[i]);
            }
        }

    
        for(let k = 0; k < inputSize; k++){
            errorVectorL2[k] = ((Math.pow((targetVector[k] - transformedVector[k]),2))/2);
        }


        for(let i = 0; i < inputSize; i++){
            for(let j = 0; j < inputSize; j++){
                dErrorVectr2[i][j] = (targetVector[i] - transformedVector[i]);
            }
        }


        for(let k = 0; k < inputSize; k++){
            errorInitial = errorInitial + ((Math.pow((targetVector[k] - transformedVector[k]),2))/2);
        }

        //Backpropogate Weight Matrix 2
        weightMatrix2 = backPropagateO(weightMatrix2, dErrorVectr2, outputVectorA1, dOutput2);

        //Transpose Weight Matrix 2
        for(let s = 0; s < inputSize; s++){
            for(let t = 0; t < inputSize; t++){
                weightMatrix2Transpose[t][s] = weightMatrix2[s][t];
            }
        }

        //Backpropogate Weight Matrix 1
        weightMatrix1 = backPropagateH1(weightMatrix1, outputVector1, dOutput1, outputVectorA1);

        console.log("Error: "+errorInitial.toExponential());
    }
}
}

}

const nn1 = new ResNN();
nn1.train();