/************************************************************************/
/*                                                                      */
/*      Copyright 2009-2011 by Joachim Schleicher and Sven Ebser        */
/*                                                                      */
/*    This file is part of the VIGRA computer vision library.           */
/*    The VIGRA Website is                                              */
/*        http://hci.iwr.uni-heidelberg.de/vigra/                       */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/************************************************************************/

#ifndef POISSON_SOLVER_HXX
#define POISSON_SOLVER_HXX

#include <iostream>
#include <vigra/stdimage.hxx>
#include <vigra/convolution.hxx>
#include <vigra/stdconvolution.hxx>
#include <vigra/basicgeometry.hxx>

#define MODE_V_CYCLE 1
#define MODE_W_CYCLE 2

namespace vigra {
/**
 * \addtogroup PDE-Solver Multigrid-Solver für Partielle Differentialgleichungen
 * 
 * \author Sven Ebser und Joachim Schleicher
 * \date Oktober 2009 - März 2010
 * 
 * \brief In diesem Projekt sollen partielle Differentialgleichungen,
 * hier speziell eine Poisson-Gleichung iterativ mit einem Multigrid-Ansatz gelöst werden.
 * 
 * Die effektive Lösung von Differentialgleichungen ist in vielen Bereichen der 
 * Bildverarbeitung von Bedeutung. In unserem Projekt ging es um die Komprimierung von 
 * HDR-Fotos zur Darstellung auf herkömmlichen 8-bit Ausgabegeräten.
 * 
 * Ein iterativer Ansatz zur Lösung der Poisson-Gleichung Δx = b ist der Algorithmus der 
 * Successive Over-Relaxation. Dieser Algorithmus wurde hier implementiert und 
 * zur Ausnutzung der Glättung hochfrequenter Anteile ein Multigrid-Code darauf aufbauend 
 * geschrieben.
 * 
 * 
 * \file poissonSolver.hxx
 * \brief Enthält den SOR- und den Multigrid-Solver
 * 
 * Diese Datei enthält den kompletten Multigrid-Code und kann einfach inkludiert werden.
 * Als Beispielanwendung wird eine Funktion zur Integration eines Gradientenfeldes angeboten.
 */


/**
 * Discrete Laplace Scharr Filter
 *
 */
template <class ARITHTYPE>
class Kernel2DLaplace : public Kernel2D<ARITHTYPE> {
public:
    Kernel2DLaplace() {
        this->initExplicitly(vigra::Diff2D(-1,-1), vigra::Diff2D(1,1)) =
             0.1878, 0.6244,  0.1878,
             0.6244,-3.2488,  0.6244,
             0.1878, 0.6244,  0.1878 ;
        this->setBorderTreatment(vigra::BORDER_TREATMENT_REFLECT);
    }
};

// Berechnet das Kernel-Produkt von a_ij mit x zur Verwendung in der Summe
// Mit Randueberpruefung
// internal use only
template <class Image>
inline typename Image::value_type a_ij_x_product(const int x0, const int y0, int dx, int dy, const Kernel2D<typename Image::value_type> & k, const Image& im) {
    int xj = x0+dx;
    int yj = y0+dy;

    if(xj < 0 || xj >= im.width())  dx = -dx;   // Wenn wir links raus laufen, Spiegeln wir an der Kante
    if(yj < 0 || yj >= im.height())  dy = -dy;

    xj = x0+dx;
    yj = y0+dy;

    return k(dx,dy)*im[yj][xj];

}

/**
 * Berechnet die Summe über den Stencil an einem Bildpunkt mit Randüberprüfung.
 * Wird als Teil der Iteration von runSORSolver intern verwendet.
 * Stencil wird gespiegelt, also BORDER_TREATMENT_REFLECT
 */
template <class Image>
inline typename Image::value_type stencil_sum( const int x, const int y, const Kernel2D<typename Image::value_type>& s, const Image& im) {
    typename Image::value_type sum =
                    a_ij_x_product(x,y,1,0,s,im) +  // spaetere Pixel (weiter rechts bzw. unten)
                    a_ij_x_product(x,y,1,1,s,im) +
                    a_ij_x_product(x,y,0,1,s,im) +
                    a_ij_x_product(x,y,-1,1,s,im) +
                    a_ij_x_product(x,y,-1,0,s,im) +   // vorherige Pixel
                    a_ij_x_product(x,y,-1,-1,s,im) +
                    a_ij_x_product(x,y,0,-1,s,im) +
                    a_ij_x_product(x,y,1,-1,s,im);      // 3x3-Stencil
    return sum;
}


// Gaussian reduction to next pyramid level (factor 2)
// intenal use in runMultigrid
// out must have size((in.width()+1)/2, (in.height()+1/2)) !!
template <class Image>
void reduceToNextLevel(Image & in, Image & out)
{
    // image size at current level
    int width = in.width();
    int height = in.height();

    // define a smoothing kernel (size 3x1)
    vigra::Kernel1D<double> filter;
    filter.initExplicitly(-1, 1) = 0.25, 0.5, 0.25;

    vigra::BasicImage<typename Image::value_type> tmpimage1(width, height);
    vigra::BasicImage<typename Image::value_type> tmpimage2(width, height);

    // smooth (band limit) input image
    separableConvolveX(srcImageRange(in),
                       destImage(tmpimage1), kernel1d(filter));
    separableConvolveY(srcImageRange(tmpimage1),
                       destImage(tmpimage2), kernel1d(filter));

    // downsample smoothed image
    resizeImageNoInterpolation(srcImageRange(tmpimage2), destImageRange(out));

    transformImage(srcImageRange(out), destImage(out), functor::Param(4.)*functor::Arg1());
}

/**
 * Konstruiert eine Gauß-Pyramide mit korrekter Skalierung:
 * Von einer Ebene zur nächstgröberen werden folgende Operationen angewendet:<br>
 * - Glättung mit Gauss-Filter<br>
 * - subsampling
 * - Multiplikation der Werte mit Faktor 4, damit die Skalierung erhalten bleibt (Gradienten...)
 * \param p[in,out] Pyramide mit Original auf Ebene 0
 * \param s Breite des Gauss. Default-Wert: 1.3
 */
template <class Image>
void constructGaussPyramid(ImagePyramid<Image>& p, float s=1.3) {
    for(int i=0; i<p.size()-1; i++) {
        Image help(p[i].width(), p[i].height());
        gaussianSmoothing(srcImageRange(p[i]), destImage(help), s);
        resampleImage(srcImageRange(help), destImage(p[i+1]), 0.5);
        transformImage(srcImageRange(p[i+1]),destImage(p[i+1]), functor::Arg1()*functor::Param(4.0));
    }
}

/**
 * Erster Schritt des Multigrid: Konstruktion eines Startbildes.
 * Multilevel-Ansatz vom gröbsten zum feinsten Gitter.
 * \param[out] out Ergebnis-Bild
 * \param[in] gradDiv Rechte Seite der Poisson-Gleichung
 * \param[in] kernel, wenn ein anderer als ein Laplace-Kernel übergeben wird, ist es keine Poisson-Init-Funktion mehr.
 * \param minLen Minimale Länge der kürzeren Seite auf dem gröbsten Gitter
 * \param omega Parameter Omega des SOR-Algorithmus. Werte zwischen 0 und 2 werden akzeptiert.
 */
template <class Image>
void poissonInitUpwards(Image& out, const Image& gradDiv, const Kernel2D<typename Image::value_type>& kernel, int minLen, float omega=1.6, float errorThreshold=1.) {
    int width = out.width();
    int height = out.height();
    int nlevels = (int)ceil(log(std::min(width, height)/(double)minLen)/log(2.));

    ImagePyramid<Image> pDiv(0,nlevels-1,gradDiv);  // Pyramid of right hand side
    ImagePyramid<Image> pRes(0,nlevels-1,out);      // Result images

    constructGaussPyramid(pDiv, 1.3);  // Gauss-filter and subsampling
    resizeImageLinearInterpolation(srcImageRange(out), destImageRange(pRes.back())); // initial Guess
    for(int i = pRes.size()-1; i > 0; i--) {
        runMultigrid(pRes[i], pDiv[i], kernel, minLen, omega, 1., MODE_V_CYCLE);
        resizeImageNoInterpolation(srcImageRange(pRes[i]), destImageRange(pRes[i-1]));
    }
    copyImage(srcImageRange(pRes[0]), destImage(out));

}

/**
 * \brief Integriert ein Gradientenfeld durch Lösung einer Poisson-Gleichung nach der Multigrid-Methode.
 *
 * Das Problem, ein Gradientenbild G=(gradx, grady) zu integrieren kann auf
 * ein Variationsproblem zurückgeführt werden, das eine Poissongleichung ergibt.
 * Diese Poisson-Gleichung wird hier mittels
 * Multigrid-Algorithmus löst die Poisson-Gleichung Δx = b iterativ mit dem gegebenen Bild <tt>ziel</tt>
 * als Startwert.
 *
 * \param[in,out] ziel Gleichzeitig Startwert (Initial Guess) und Rückgabebild für die Lösung.
 *                  Diese Werte werden iterativ verbessert.
 * \param[in] gradx x-Ableitung des gesuchten Bildes.
 *              Wird vom Algorithmus nicht verändert.
 * \param[in] grady y-Ableitung des gesuchten Bildes. Wird nicht verändert.
 * \param minLen Parameter des Multigrid-Algorithmus. Siehe runMultigrid()
 * \param omega Parameter Omega des SOR-Algorithmus.  Siehe runSORSolver()
 * \param error Parameter des Multigrid-Algorithmus. Siehe runMultigrid()
 */
template <class Image>
void integrateGradientFieldMultigrid(Image& out, Image& gradx, Image& grady, int minLen=33, double omega=1.7, float errorThreshold=1.) {
        vigra_precondition(out.width()==gradx.width()&&out.height()==gradx.height(),"wrong dimensions to integrate");
        Kernel1D<typename Image::value_type> derivative;
        derivative.initExplicitly(-1,0) =  1, -1;
        int width = out.width();
        int height = out.height();
        derivative.setBorderTreatment(BORDER_TREATMENT_AVOID); // Rand extra
        Image gradxx(width, height), gradyy(width,height), zielDiv(width,height);

        // Die Randwerte sind (-2)*(erste Ableitung) und werden wegen
        // BORDER_TREATMENT_AVOID nicht mehr veraendert
        vigra::transformImage(srcImageRange(gradx),
                            destImage(gradxx),
                            functor::Arg1()*functor::Param(-2.));     // Warum eigentlich -2??
        vigra::transformImage(srcImageRange(grady),
                            destImage(gradyy),
                            functor::Param(-2.)*functor::Arg1());

        separableConvolveX(srcImageRange(gradx),
                           destImage(gradxx), kernel1d(derivative));
        separableConvolveY(srcImageRange(grady),
                           destImage(gradyy), kernel1d(derivative));
        vigra::combineTwoImages(
            srcIterRange(gradxx.upperLeft(), gradxx.lowerRight()),
            srcIter(gradyy.upperLeft()),
            destIter(zielDiv.upperLeft()),
            functor::Arg1()+functor::Arg2());

        Kernel2D<typename Image::value_type> laplaceKernel = Kernel2DLaplace<typename Image::value_type>();
        poissonInitUpwards(out, zielDiv, laplaceKernel, minLen);
        runMultigrid(out, zielDiv, laplaceKernel, minLen, omega, errorThreshold);

        // Fehler berechnen
        Image err(width, height);
        double _error = poissonError(err, out, zielDiv, laplaceKernel);
        std::cout << "Fehler: " << _error << std::endl;

}

/**
 * \brief Multigrid-Verfahren zur Lösung der Poisson-Gleichung
 * mit von Neumann-Randbedingungen.
 *
 * Der Algorithmus löst die Poisson-Gleichung Δx = b iterativ mit dem gegebenen Bild <tt>ziel</tt>
 * als Startwert. Auf unterschiedlich feinen Stufen wird jeweils der Self-over-relaxation-Algorithmus
 * als Glätter verwendet.
 *
 * \param[in,out] ziel Gleichzeitig Startwert (Initial Guess) und Rückgabebild für die Lösung.
 *                  Diese Werte werden iterativ verbessert.
 * \param[in] gradient Rechte Seite <tt>b</tt> der Poisson-Gleichung.
 *              Wird vom Algorithmus nicht verändert.
 * \param minLen Minimale Länge der kürzeren Seite auf dem gröbsten Gitter
 * \param omega Parameter Omega des SOR-Algorithmus. Werte zwischen 0 und 2 werden akzeptiert.
 * \param errorThreshold Wahl der Genauigkeit. Hiermit wird die Abbruchbedingung skaliert.
 * \param mode MODE_V_CYCLE oder MODE_W_CYCLE
 */
template <class Image>
void runMultigrid(Image& out, Image& gradient, const Kernel2D<typename Image::value_type>& k, int minLen, typename Image::value_type omega, float errorThreshold=1., int mode=MODE_W_CYCLE) {
    int width = out.width();
    int height= out.height();

    if(width < minLen || height < minLen) {  // groebstes Gitter erreicht
        //~ std::cout << "Iteration beendet" << std::endl;
        return;
    }
    //~ std::cout << "wxl: " << width << "x" << height << std::endl;
    Image err(width, height);
    Image err2((width+1)/2,(height+1)/2);
    Image out2((width+1)/2,(height+1)/2);

    runSORSolver(out, gradient, k, omega, 0.05*errorThreshold);  // Vorglättung

    poissonError(err, out, gradient, k);    // Fehler berechnen

    reduceToNextLevel(err, err2);
    runMultigrid(out2, err2, k, minLen, omega, errorThreshold);
    if(mode==MODE_W_CYCLE) {
        runMultigrid(out2, err2, k, minLen, omega, errorThreshold); // again: W-Cycle
    }

    vigra::resizeImageLinearInterpolation(srcImageRange(out2), destImageRange(err)); // re-use err-Image

    // Nachglaettung auch hier moeglich mit err als rechter Seite.
    vigra::combineTwoImages(                        // x_neu=ziel+korrektur
        srcImageRange(out),
        srcImage(err),
        destImage(out),
        functor::Arg1()-functor::Arg2());
    runSORSolver(out, gradient, k, 1., 0.02*errorThreshold);    // Nachglättung mit Gauss-Seidel->omega=1.0
    return;
}


/**
 * \brief Self-over-relaxation zur Lösung der Poisson-Gleichung
 * mit von Neumann-Randbedingungen.
 *
 * Der Algorithmus löst die Poisson-Gleichung Δx = b iterativ mit dem gegebenen Bild <tt>ziel</tt>
 * als Startwert.
 *
 * \param[in,out] ziel Gleichzeitig Startwert (Initial Guess) und Rückgabebild für die Lösung.
 *                  Diese Werte werden iterativ verbessert.
 * \param[in] gradient Rechte Seite <tt>b</tt> der Poisson-Gleichung.
 *              Wird vom Algorithmus nicht verändert.
 * \param omega Parameter Omega des SOR-Algorithmus. Werte zwischen 0 und 2 werden akzeptiert.
 */
template <class Image>
void runSORSolver(Image& ziel, const Image& gradient, const Kernel2D<typename Image::value_type>& s, typename Image::value_type omega=1.6, float errorThreshold=10) {
    typedef typename Image::value_type pixelType;
    int width = ziel.width();
    int height= ziel.height();

    pixelType oldError = 0;         // Aenderung in voriger Iteration
    for(int j = 0; j < 50; j++) {   // Iterationen, obere Schranke
        pixelType error=0;
        // aus Effizienzgründen werden Randpixel separat behandelt, da der Stencil
        // dort gespiegelt werden muss
        for(int i = 0; i < width; i++) {    // erste Zeile

            pixelType xnew = (1-omega)*ziel[0][i] + omega * (gradient[0][i] - stencil_sum(i,0,s,ziel))/s(0,0);
            pixelType delta = ziel[0][i]-xnew;
            ziel[0][i] = xnew;
            error += delta*delta;
        }
        for(int y = 1; y < height-1; y++) { // Ueber das ganze Bild
            { // linker Rand
                pixelType xnew = (1-omega)*ziel[y][0] + omega * (gradient[y][0] - stencil_sum(0,y,s,ziel))/s(0,0);
                pixelType delta = ziel[y][0]-xnew;
                ziel[y][0] = xnew;
                error += delta*delta;
            }
            for(int x = 1; x < width-1; x++) { // Mitte, zeitkritischer Teil
                pixelType sum =     s(1,0)*ziel[y+1][x] +  // spaetere Pixel (weiter rechts bzw. unten)
                                    s(1,1)*ziel[y+1][x+1] +
                                    s(0,1)*ziel[y][x+1] +
                                    s(-1,1)*ziel[y-1][x+1] +
                                    s(-1,0)*ziel[y-1][x] +
                                    s(-1,-1)*ziel[y-1][x-1] +
                                    s(0,-1)*ziel[y][x-1] +
                                    s(1,-1)*ziel[y+1][x-1];
                pixelType xnew = (1-omega)*ziel[y][x] + omega * (gradient[y][x] - sum)/s(0,0);
                pixelType delta = ziel[y][x]-xnew;
                error += delta*delta;
                ziel[y][x] = xnew;
            }
            { // rechter Rand
                pixelType xnew = (1-omega)*ziel[y][width-1] + omega * (gradient[y][width-1] - stencil_sum(width-1,y,s,ziel))/s(0,0);
                pixelType delta = ziel[y][width-1]-xnew;
                ziel[y][width-1] = xnew;
                error += delta*delta;
            }
        }
        for(int x = 0; x < width; x++) {    // letzte Zeile

            pixelType xnew = (1-omega)*ziel[height-1][x] + omega * (gradient[height-1][x] - stencil_sum(x,height-1,s,ziel))/s(0,0);
            pixelType delta = ziel[height-1][x]-xnew;
            ziel[height-1][x] = xnew;
            error += delta*delta;
        }



        if (1) {   // jeden x. Schritt ausgeben
            //~ std::cout << "Schritt " << j << " mit Aenderung " << 1000*error/(double)(width*height) << "  rel :" << 1000*(oldError-error)/error << std::endl;
            //~ std::cout << error << std::endl;
            //~ std::cout << 1000*log(oldError/error)/log(10.) << std::endl;
        }
        if(oldError>0 && log(oldError/error)/log(10.)<errorThreshold) {  // Fertig
            //~ std::cout << "Fehlerschranke " << errorThreshold << " erreicht nach " << j << " Schritten. Ende" << std::endl;
            break;
        }
        oldError = error;
    }
    //~ std::cout << std::endl;

    return;


}

/**
 * \brief Residuum der Poisson-Gleichung berechnen
 *
 * Die Funktion berechnet das Residuum <tt>error=Ax-b</tt>, wenn x eine
 * Näherungslösung der Poisson-Gleichung Δx = b ist.
 *
 * \param[out] error residuum
 * \param[in]  ziel  Näherungslösung <tt>x</tt>
 * \param[in]  gradient Rechte Seite <tt>b</tt>
 * \param[in]  kernel Kernel, der dem Differentialoperator der PDE entspricht <tt>b</tt>
 * \return Summe über die Fehlerquadrate
 */
template <class Image>
double poissonError(Image& error, const Image& ziel, const Image& gradient, const Kernel2D<typename Image::value_type> kernel) {
    vigra::convolveImage(srcImageRange(ziel), destImage(error), kernel2d(kernel)); // A*x
    vigra::combineTwoImages(                        // r = A*x-b
        srcImageRange(error),
        srcImage(gradient),
        destImage(error),
        functor::Arg1()-functor::Arg2());
    Image _error(ziel.width(), ziel.height());
    vigra::transformImage(srcImageRange(error),     // Quadratisch
                  destImage(_error),
                  functor::Arg1()*functor::Arg1() );
    vigra::FindSum<typename Image::PixelType> sum;  // ||r||^2
    vigra::inspectImage(srcImageRange(_error), sum);

    return sum.sum();
}

} // namespace vigra

#endif //POISSON_SOLVER_HXX
