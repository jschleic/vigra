#include <iostream>

#include <vigra/timing.hxx>
#include <vigra/impex.hxx>
#include <vigra/convolution.hxx>
#include <vigra/stdconvolution.hxx>

#include "poissonSolver.hxx"

// This can be used as usage example or as testcase
// It loads an image given as commandline parameter
// calculates the divergence and passes the divergence
// to the solver to recover the original image.
int main(int argc, char** argv) {
    if(argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " infile outfile" << std::endl;
        std::cout << "(supported formats: " << vigra::impexListFormats() << ")" << std::endl;
        return 1;
    }

    try
    {
        USETICTOC;
        TIC;

        vigra::ImageImportInfo info(argv[1]);

        vigra_precondition(info.isGrayscale(), "Sorry, cannot operate on color images");

        int width = info.width();
        int height = info.height();

        vigra::FImage in(width, height);
        importImage(info, destImage(in)); //Eingabe Bild

        vigra::FImage zielDiv(width, height);

        vigra::Kernel2D<float> s;
        s.initExplicitly(vigra::Diff2D(-1,-1), vigra::Diff2D(1,1)) =
         0.1878, 0.6244,  0.1878,
         0.6244,-3.2488,  0.6244,
         0.1878, 0.6244,  0.1878 ;
        s.setBorderTreatment(vigra::BORDER_TREATMENT_REFLECT);

        vigra::convolveImage(srcImageRange(in), destImage(zielDiv), kernel2d(s)); // A*x
        //~ exportImage(srcImageRange(zielDiv), vigra::ImageExportInfo("ziel.png"));

        // wieder integrieren, also Poisson-Glg loesen
        vigra::FImage out(width, height);
        vigra::FImage err(width, height);
        vigra::FImage tempDiv(width, height);

        vigra::Kernel2D<float> laplaceKernel = vigra::Kernel2DLaplace<float>();
        vigra::poissonInitUpwards(out, zielDiv, laplaceKernel, 33);
        vigra::runMultigrid(out, zielDiv, laplaceKernel, 33, 1.7, 1.0);

        // Fehler berechnen
        double _error = poissonError(err, out, zielDiv, laplaceKernel);
        std::cout << "Fehler (Residuumsquadrat): " << _error << std::endl;


        exportImage(srcImageRange(out), vigra::ImageExportInfo(argv[2]));//Ausgabe des Bildes

        TOC;
    }
    catch (vigra::StdException & e)
    {
        std::cout << e.what() << std::endl;
        std::cout<<"An error occured. Terminating."<<std::endl;
        return 1;
    }


}
