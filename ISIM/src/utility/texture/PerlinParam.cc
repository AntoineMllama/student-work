
#include "PerlinParam.hh"
#include <iostream>
#include <cmath>

PerlinParam::PerlinParam(std::string type)
:type(type)
{}

double linear_interpolation(double t, double a, double b) {
    return a + t * (b - a);
}

double lissage_quintique(double t) {
    double t3 = t * t * t;
    return t3 * (t * (t * 6. - 15.) + 10.);
}


// -------------------------------------------------------------------------------------------------------
// Programme Get2DPerlinNoiseValue (Bois 1, Bois 2, Marbre 1)


double Get2DPerlinNoiseValue(double x, double y, double res)
{
    double tempX,tempY;
    int x0,y0,ii,jj,gi0,gi1,gi2,gi3;
    double tmp,s,t,u,v,Cx,Cy,Li1,Li2;
    double gradient[][2] = {{1,1},{-1,1},{1,-1},{-1,-1},{1,0},{-1,0},{0,1},{0,-1}};

    int perm[] =
       {151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,
        142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,
        203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
        74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,
        105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,
        187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,
        64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,
        47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,
        153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,
        112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,
        235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,
        127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,
        156,180};

    x /= res;
    y /= res;

    x0 = (int)(x);
    y0 = (int)(y);

    ii = x0 & 255;
    jj = y0 & 255;

    gi0 = perm[ii + perm[jj]] % 8;
    gi1 = perm[ii + 1 + perm[jj]] % 8;
    gi2 = perm[ii + perm[jj + 1]] % 8;
    gi3 = perm[ii + 1 + perm[jj + 1]] % 8;

    tempX = x-x0;
    tempY = y-y0;
    s = gradient[gi0][0]*tempX + gradient[gi0][1]*tempY;

    tempX = x-(x0+1);
    tempY = y-y0;
    t = gradient[gi1][0]*tempX + gradient[gi1][1]*tempY;

    tempX = x-x0;
    tempY = y-(y0+1);
    u = gradient[gi2][0]*tempX + gradient[gi2][1]*tempY;

    tempX = x-(x0+1);
    tempY = y-(y0+1);
    v = gradient[gi3][0]*tempX + gradient[gi3][1]*tempY;


    tmp = x-x0;
    Cx = lissage_quintique(tmp);


    Li1 = linear_interpolation(Cx, s, t);
    Li2 = linear_interpolation(Cx, u, v);

    tmp = y - y0;
    Cy = lissage_quintique(tmp);


    return linear_interpolation(Cy, Li1, Li2);
}


// Bois 1:
void PerlinParam::bois1(Size& size) {
    sf::Image image;
    int width = size.width;
    int height = size.height;
    image.create(width, height);

    double bruit;
    for (double x = 0; x < width; x++) {
        for (double y = 0; y < height; y++) {
            bruit = Get2DPerlinNoiseValue(x, y, 50);

            double seuil = 0.2;

            double valeur = fmod(bruit, seuil);
            if(valeur > seuil / 2)
                valeur = seuil - valeur;

            double f = (1 - cos(M_PI * valeur / (seuil / 2))) / 2;

            double r = 0.6 * (1 - f) + 0.3 * f;
            double g = 0.4 * (1 - f) + 0.2 * f;
            double b = 0.1 * (1 - f) + 0.05 * f;

            r = r * 255;
            g = g * 255;
            b = b * 255;


            sf::Color color;
            sf::Uint8 red = r;
            sf::Uint8 green = g;
            sf::Uint8 blue = b;

            color = sf::Color(red, green, blue);

            image.setPixel(x, y, color);
        }
    }
    this->image = image;
}


// Bois 2:
void PerlinParam::bois2(Size& size) {
    sf::Image image;
    int width = size.width;
    int height = size.height;
    image.create(width, height);

    double bruit = 0.0;
    for (double x = 0; x < width; x++) {
        for (double y = 0; y < height; y++) {
            bruit = Get2DPerlinNoiseValue(x, y, 100);
            double val = (bruit + 1) / 2;
            val = 9 * val;
            double f = val - std::round(val);
            f = f * 255;
            double r = 139 + f * 0.5;
            double g = 69 + f * 0.5;
            double b = 19 + f * 0.5;

            if (r < 0 || g < 0 || b < 0)
                b = 0;

            sf::Color color;
            sf::Uint8 red = r;
            sf::Uint8 green = g;
            sf::Uint8 blue = b;

            color = sf::Color(red, green, blue);

            image.setPixel(x, y, color);
        }
    }
    this->image = image;
}


// Marbre 1:
void PerlinParam::marbre1(Size& size) {
    sf::Image image;
    int width = size.width;
    int height = size.height;
    image.create(width, height);

    double bruit = 0.0;
    for (double x = 0; x < width; x++) {
        for (double y = 0; y < height; y++) {
            bruit = Get2DPerlinNoiseValue(x, y, 100);

            double f = 1 - sqrt(fabs(sin(2 * M_PI * bruit)));

            double r = 1.0 * (1 - f) + 0.5 * f;
            double g = 1.0 * (1 - f) + 0.5 * f;
            double b = 1.0 * (1 - f) + 0.5 * f;

            r = r * 255;
            g = g * 255;
            b = b * 255;


            sf::Color color;
            sf::Uint8 red = r;
            sf::Uint8 green = g;
            sf::Uint8 blue = b;

            color = sf::Color(red, green, blue);

            image.setPixel(x, y, color);
        }
    }
    this->image = image;
}

// -------------------------------------------------------------------------------------------------------
// Programme MarblePerlin (Marbre 2, Ciel, Bois 3)

double grad(int hash, double x, double y) {
    int gradientIndex = hash & 15;

    double u = y;
    double v = 0;

    if (gradientIndex < 8) {
        u = x;
    }

    if (gradientIndex < 4) {
        v = y;
    } else if (gradientIndex == 12 || gradientIndex == 14) {
        v = x;
    }

    double gradU = (gradientIndex & 1) ? -u : u;
    double gradV = (gradientIndex & 2) ? -v : v;

    return gradU + gradV;
}

void generatePermutation(std::vector<int>& p) {
    for (int i = 0; i < 256; ++i)
        p[i] = i;
    for (int i = 255; i > 0; --i) {
        int j = rand() % (i + 1);
        std::swap(p[i], p[j]);
    }
    for (int i = 0; i < 256; ++i)
        p[256 + i] = p[i];
}

double perlinNoise(double x, double y, const std::vector<int>& p) {
    int X = static_cast<int>(floor(x)) & 255;
    int Y = static_cast<int>(floor(y)) & 255;

    double xDec = x - floor(x);
    double yDec = y - floor(y);

    double u = lissage_quintique(xDec);
    double v = lissage_quintique(yDec);

    int A = p[X] + Y;
    int AA = p[A];
    int AB = p[A + 1];
    int B = p[X + 1] + Y;
    int BA = p[B];
    int BB = p[B + 1];

    double gradAA = grad(p[AA], xDec, yDec);
    double gradBA = grad(p[BA], xDec - 1, yDec);
    double gradAB = grad(p[AB], xDec, yDec - 1);
    double gradBB = grad(p[BB], xDec - 1, yDec - 1);

    double lerpU_A = linear_interpolation(u, gradAA, gradBA);
    double lerpU_B = linear_interpolation(u, gradAB, gradBB);
    double result = linear_interpolation(v, lerpU_A, lerpU_B);

    return result;
}

double perlin(double x, double y, const std::vector<int>& p) {
    double value = 0.0;
    double amplitude = 1.0;
    double frequency = 1.0;
    for (int i = 0; i < 6; ++i) {
        value += perlinNoise(x * frequency, y * frequency, p) * amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return value;
}


// Marbre 2:
void PerlinParam::marbre2(Size& size) {
    sf::Image image;
    int width = size.width;
    int height = size.height;
    image.create(width, height);

    std::vector<int> permutation(512);
    generatePermutation(permutation);

    for (double x = 0; x < width; x++) {
        for (double y = 0; y < height; y++) {
            double nx = double(x) / width;
            double ny = double(y) / height;
            double noiseValue = perlin(nx, ny, permutation);
            double marbleValue = 1.0 - fabs(sin(2 * M_PI * noiseValue));
            marbleValue = pow(marbleValue, 4.0);

            double r = linear_interpolation(marbleValue, 0.9, 0.4);
            double g = linear_interpolation(marbleValue, 0.9, 0.4);
            double b = linear_interpolation(marbleValue, 0.9, 0.4);

            r = r * 255;
            g = g * 255;
            b = b * 255;

            sf::Color color;
            sf::Uint8 red =  r;
            sf::Uint8 green = g;
            sf::Uint8 blue =  b;

            color = sf::Color(red, green, blue);

            image.setPixel(x, y, color);
        }
    }
    this->image = image;
}

// Bois 3:
void PerlinParam::bois3(Size& size) {
    sf::Image image;
    int width = size.width;
    int height = size.height;
    image.create(width, height);

    std::vector<int> permutation(512);
    generatePermutation(permutation);

    for (double x = 0; x < width; x++) {
        for (double y = 0; y < height; y++) {
            double nx = double(x) / width;
            double ny = double(y) / height;
            double noise = perlin(nx, ny, permutation);
            double noiseValue = noise / 0.1;
            noiseValue = fabs(sin(noiseValue * M_PI));

            double r = this->bois3_mid.r;
            double g = this->bois3_mid.g;
            double b = this->bois3_mid.b;

            if (noiseValue < this->borne_inf) {
                r = this->bois3_inf.r;
                g = this->bois3_inf.g;
                b = this->bois3_inf.b;
            } else if (noiseValue > borne_sup) {
                r = this->bois3_up.r;
                g = this->bois3_up.g;
                b = this->bois3_up.b;
            }

            sf::Color color;
            sf::Uint8 red =  r;
            sf::Uint8 green = g;
            sf::Uint8 blue =  b;

            color = sf::Color(red, green, blue);

            image.setPixel(x, y, color);
        }
    }
    this->image = image;
}


// Ciel:
void PerlinParam::ciel(Size& size) {
    sf::Image image;
    int width = size.width;
    int height = size.height;
    image.create(width, height);


    std::vector<int> permutation(512);
    generatePermutation(permutation);


    for (double x = 0; x < width; x++) {
        for (double y = 0; y < height; y++) {

            double nx = double(x) / width;
            double ny = double(y) / height;
            double noiseValue = perlin(nx, ny, permutation);
            double marbleValue = 0.5 * (1.0 + sin(2.0 * M_PI * noiseValue));


            double r = linear_interpolation(marbleValue, 1, 0.2);
            double g = linear_interpolation(marbleValue, 1, 0.6);
            double b = linear_interpolation(marbleValue, 1, 1);

            r = r * 255;
            g = g * 255;
            b = b * 255;

            sf::Color color;
            sf::Uint8 red =  r;
            sf::Uint8 green = g;
            sf::Uint8 blue =  b;

            color = sf::Color(red, green, blue);

            image.setPixel(x, y, color);
        }
    }
    this->image = image;
}



void PerlinParam::run(Size& size, std::string saveAs) {
    if (this->type == "bois1") {
        bois1(size);
    }
    else if (this->type == "bois2") {
        bois2(size);
    }
    else if (this->type == "marbre1") {
        marbre1(size);
    }
    else if (this->type == "marbre2") {
        marbre2(size);
    }
    else if (this->type == "ciel") {
        ciel(size);
    }
    else if (this->type == "bois3") {
        bois3(size);
    }
    else {
        exit(-1);
    }

    if (!saveAs.empty()) {
        if (!this->image.saveToFile(saveAs)) {
            std::cout << "SaveAs Fail with : " + saveAs + '\n';
        }
    }
}
