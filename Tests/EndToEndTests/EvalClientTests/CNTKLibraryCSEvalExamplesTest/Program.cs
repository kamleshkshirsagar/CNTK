using System;
using CNTK;

namespace CNTKLibraryCSEvalExamples
{
    class program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("======== Evaluate model using C# ========");

            CNTKLibraryManagedExamples.EvaluationSingleImage(DeviceDescriptor.CPUDevice);
            CNTKLibraryManagedExamples.EvaluationBatchOfImages(DeviceDescriptor.CPUDevice);
            //TODO: Add examples with OneHot.
            //EvaluationSingleSequenceUsingOneHot(DeviceDescriptor.CPUDevice);
            //EvaluationBatchOfSequencesUsingOneHot(DeviceDescriptor.CPUDevice);

            // TODO: using GPU.
            //EvaluationSingleImage(DeviceDescriptor.GPUDevice(0));
            //EvaluationBatchOfImages(DeviceDescriptor.GPUDevice(0));

            Console.WriteLine("======== Evaluation completes. ========");
        }
    }
}
