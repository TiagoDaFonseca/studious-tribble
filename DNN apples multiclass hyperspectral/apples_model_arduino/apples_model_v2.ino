#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/version.h"
#include <stdio.h>

//import model
#include "apples_model.h"

#define DEBUG 1

//define tflite namespace
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;
  constexpr int kTensorArenaSize = 5 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
  }

// Other global variables
int inference_count = 1;

void setup() {
  // put your setup code here, to run once:
#if DEBUG
    while(!Serial);
#endif
  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(apples_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while(1);
  }

  // Pull in only needed operations (should match NN layers)
  // Available ops:
  //  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/micro_ops.h
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while(1);
  }

  // Assign model input and output buffers (tensors) to pointers
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);

  // Get information about the memory area to use for the model's input
  // Supported data types:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h#L226
#if DEBUG
  Serial.print("Number of dimensions: ");
  Serial.println(model_input->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(model_input->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(model_input->dims->data[1]);
  Serial.print("Input type: ");
  Serial.println(model_input->type);
  Serial.print("Number of dimensions: ");
  Serial.println(model_output->dims->size);
  Serial.print("Dim 1 size: ");
  Serial.println(model_output->dims->data[0]);
  Serial.print("Dim 2 size: ");
  Serial.println(model_output->dims->data[1]);
  Serial.print("Input type: ");
  Serial.println(model_output->type);
#endif
}

void loop() {
  // put your main code here, to run repeatedly:
  #if DEBUG
  unsigned long start_timestamp = millis();
#endif
  //Create a spectrum to feed the model
  float x_val[204] = {0.366666675, 0.333333343, 0.280000001, 0.253968269, 0.239999995, 0.218390808, 0.222222224, 0.207207203, 0.203252032, 0.2074074, 0.201342285, 0.196319014, 0.196629211, 0.195876285, 0.193396226, 0.192139745, 0.19758065, 0.200000003, 0.199312717, 0.203821659, 0.207715139, 0.207282916, 0.206349209, 0.208542719, 0.213942304, 0.211009175, 0.217105269, 0.220588237, 0.218875498, 0.219230771, 0.22324723, 0.224778756, 0.225255966, 0.22660099, 0.231378764, 0.2324159, 0.232352942, 0.230878189, 0.234652117, 0.231979027, 0.237074405, 0.240875915, 0.238823533, 0.242009133, 0.245283023, 0.24406047, 0.243414119, 0.245901644, 0.246246248, 0.24583742, 0.248803824, 0.248826295, 0.25, 0.250678748, 0.251561105, 0.253744483, 0.253472209, 0.25731498, 0.257021278, 0.255480617, 0.259631485, 0.2570481, 0.263201326, 0.260869563, 0.268074721, 0.268391281, 0.271922767, 0.27635783, 0.283439487, 0.288095236, 0.29588607, 0.302454472, 0.308846772, 0.314624518, 0.324345767, 0.334126979, 0.344470948, 0.351718634, 0.362690151, 0.372485936, 0.387278587, 0.394842863, 0.40647772, 0.418092906, 0.419196069, 0.422757477, 0.427135676, 0.429661006, 0.439862549, 0.439722449, 0.441512764, 0.439678282, 0.431425989, 0.42949906, 0.423984885, 0.426923066, 0.429268301, 0.436023623, 0.459325403, 0.493506491, 0.527472556, 0.561999977, 0.590772331, 0.611222446, 0.622983873, 0.636548221, 0.642126799, 0.653526962, 0.656512618, 0.660638273, 0.664502144, 0.668859661, 0.667781472, 0.674259663, 0.670520246  ,0.666666687  ,0.667067289, 0.663003683, 0.669987559, 0.662420392, 0.665799737, 0.667110503, 0.672554374, 0.667593896, 0.667617679, 0.673469365, 0.670640826, 0.666666687, 0.672386885, 0.674679458, 0.683774829, 0.666666687, 0.674782634, 0.673796773, 0.669104218, 0.67608285, 0.680154145, 0.672619045, 0.675510228, 0.68058455, 0.672376871, 0.676211476, 0.669662893, 0.673563242, 0.665882349, 0.669879496, 0.665841579, 0.653164566, 0.664935052, 0.65691489, 0.657608688, 0.65459609, 0.658119678, 0.656976759, 0.658753693, 0.669696987, 0.663580239, 0.661392391, 0.655844152, 0.665551841, 0.656357408, 0.657243788, 0.649819493, 0.650557637, 0.664122164, 0.64453125, 0.641129017, 0.647302926, 0.648068666, 0.64159292, 0.645454526, 0.63380283, 0.631067932, 0.623115599, 0.625, 0.618279576, 0.611111104, 0.614942551, 0.613095224, 0.602484465, 0.597402573, 0.587837815, 0.598591566, 0.602941155, 0.592307687, 0.59677422, 0.593220353, 0.566371679, 0.570093453, 0.57843136, 0.551020384, 0.569892466, 0.556818187, 0.559523821, 0.556962013, 0.560000002, 0.577464759, 0.567164183, 0.578125, 0.583333313, 0.553571403, 0.584905684, 0.579999983, 0.586956501}; // array with 204 values
  
  // Copy values to input buffer (tensor)
  for (int i=0; i<204; i++){
    model_input->data.f[i] = x_val[i];
  }
  

  // Run inference
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on input: %f\n", x_val);
  }
  //float y0 = model_output->data.f[0];
  int max_index = -1;
  float y = 0.0;
  for (int i=0; i<4; i++){
    if(model_output->data.f[i]>y){
      y = model_output->data.f[i];
      max_index = i;
    }
  }
  Serial.print("Probable class: ");
  Serial.print(max_index);
  Serial.print(" -> P= ");
  Serial.println(y);
#if DEBUG
  Serial.print("Time for inference (ms): ");
  Serial.println(millis() - start_timestamp);
#endif

  inference_count+=1;
  delay(1000);
}
