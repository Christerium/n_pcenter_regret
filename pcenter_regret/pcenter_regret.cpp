/*
 * nested_pcp.cpp
 * 
 * Created on Oct 30, 2023
 *      Author: christof
 * All 3 model + all pre runs + all primals 
 * 
 */

#include "utility/ProgramOptions.h"
#include "ilcplex/ilocplex.h"
#include <boost/filesystem.hpp>
#include <boost/timer/timer.hpp>
#include <boost/chrono.hpp>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <vector>
#include <sstream>
#include <limits>
#include <chrono>
#include <algorithm>
#include <tuple>
#include <set>
#include <map>
#include <numeric>
#include <typeinfo>
//#include <boost/system.hpp>
using namespace std;
#define epsOpt 1e-5

typedef IloArray<IloIntVarArray> IloIntVarArray2;
typedef IloArray<IloNumVarArray> IloNumVarArray2;
typedef IloArray<IloIntVarArray2> IloIntVarArray3;
//typedef IloArray<IloNumArray> IloNumArray2;

typedef boost::chrono::duration<double> sec;


struct nodeValuePair {
    int id = -1;
    int distance = -1;
    int horizon = 0;
    double value = 0.0;

    bool operator < (const nodeValuePair& other) const {
        if (distance != other.distance) {
            return distance < other.distance;
        }
        return value < other.value;
    }

    bool operator > (const nodeValuePair& other) const {
        if (distance != other.distance) {
            return distance > other.distance;
        }
        return value > other.value;
    }
};

struct nodeValuePair2 {
    int cust = -1;
    int fac = -1;
    int criticalDistance = -1;
    int horizon = 0;
    double violation = 0.0;

    bool operator > (const nodeValuePair2& other) const {
        if (violation != other.violation) {
            return violation > other.violation;
        }
        return cust > other.cust;
    }

    bool operator < (const nodeValuePair2& other) const {
        if (violation != other.violation) {
            return violation < other.violation;
        }
        return cust > other.cust;
    }
};

struct Instance {
    int nNodes;
    int startP;
    int endP;
    vector<vector<int>> distances;
    vector<pair<double, double>> coords;
    int format = 1;
    set<int> distinct_distances;
    vector<map<int, int>> bigS;
};

struct Solution {
    vector<int> y;
    vector<int> support;

    double obj = 0;

    bool changed = false;
    bool improved = false;
};

struct SolutionNested {
    vector<vector<int>> y;
    vector<int> support;

    double obj = 0;

    bool changed = false;
    bool improved = false;
};

struct cutViolation {
    IloExpr myCut;
    double violation;
    double RHS2;
};

// Instance reading
void readInstance();
void readInstanceTSP();
inline int getDistance(int i, int j);

// p-Center Models
void PC(IloModel myModel, Instance myI, IloIntVarArray2 x, IloIntVarArray y, IloIntVar z, int p);
void PCY(IloModel myModel, Instance myI, IloIntVarArray y, IloIntVar z, int p);
void PCE(IloModel myModel, Instance myI, IloIntVarArray u, IloIntVarArray y, IloIntVar R, int p);

// np-Center models
void nPC(IloModel myModel, Instance myI, IloIntVarArray3 x, IloIntVarArray2 y, IloIntVarArray z,  vector<int> P, tuple<int, int> bounds);
void nPCY(IloModel myModel, Instance myI, IloIntVarArray2 y, IloIntVarArray z, vector<int> P, tuple<int, int> bounds);
void nPCE(IloModel myModel, Instance myI, IloIntVarArray2 u, IloIntVarArray2 y, IloIntVarArray R, vector<int> P, tuple<int, int> bounds);
void nPCYRegret(IloModel myModel, Instance myI, IloIntVarArray2 y, IloIntVarArray z, IloNumVarArray w, vector<int> P, vector<int> opt);
void nPCYRegretMinMax(IloModel myModel, Instance myI, IloIntVarArray2 y, IloIntVarArray z, IloNumVar w, vector<int> P, vector<int> opt);

// Heuristics
tuple<int, int> highPHeuristic();
tuple<int, int> lowPHeuristic();
tuple<int, int> highLowHeuristic();

// Preruns

int yPreRun(IloModel myModel, int p);
int xyPreRun(IloModel myModel, int p);
int yPreRun2(IloModel myModel, int p, int LB);

// Seperation
vector<nodeValuePair2> seperationPC(IloNumArray y_lb, int LB, set<int> bigI);
vector<nodeValuePair2> seperationNpC(IloNumArray2 y_lb, vector<int> LB, int h, set<int> bigI);
cutViolation generateCut(nodeValuePair2 violatedNode, IloIntVarArray y, IloNumArray y_lb, IloEnv myEnv, int LB);
cutViolation generateNCut(nodeValuePair2 violatedNode, IloIntVarArray2 y, IloNumArray2 y_lb, IloEnv myEnv, vector<int> LB, int h);

// Fixed Customer
void initSetfC(int p);

void setCPLEXParameters(IloCplex myCplex);
void printSTATs(IloEnv masterEnv, Instance myI, vector<int> P);

int getMinMaxDistance(vector<nodeValuePair> helper);
void MIPStart(IloEnv env, IloIntVarArray2 y, IloIntVarArray z, IloCplex cplex);
void MIPStart2(IloEnv env, IloIntVarArray2 y, IloIntVarArray z, IloCplex cplex);

void MIPStart3(IloEnv env, IloIntVarArray2 y, IloIntVarArray z, IloCplex cplex, vector<IloNumArray> y_solutions);


// Global Variables
Instance myI;
//boost::timer::cpu_timer timer;
std::string filename;
Solution myIncumbent;

int userCuts = 0;
int lazyCuts = 0;
int rootUserCuts = 0;
int rootLazyCuts = 0;

int numSepRoot = 0;
int preNode = 0;

double prevLB = 0;
double preZ = 0;
int boundNotImprovedCount = 0;
int boundNotImprovedCountFixed = 0;
int nodeIterations = 0;

int maxNumCutsRoot = 100;
int maxNumCutsTree = 50;
int maxNumSepRoot = 1000;
int maxNumSepTree = 1;
int maxNoImprovements = 100;
int maxNoImprovementsFixed = 5;



double nprevLB = 0;
double npreZ = 0;
int nboundNotImprovedCount = 0;
int nboundNotImprovedCountFixed = 0;
int nnodeIterations = 0;

int nmaxNumCutsRoot = 100;
int nmaxNumCutsTree = 50;
int nmaxNumSepRoot = 1000;
int nmaxNumSepTree = 1;
int nmaxNoImprovements = 100;
int nmaxNoImprovementsFixed = 5;


int root = 1;

set<int> iHat;
set<int> inonHat;
set<int> bigI;

set<int> iHatLazy;
set<int> inonHatLazy;

int firstCuts = 0;
int secondCuts = 0;
int thirdCuts = 0;

int initLB;

int preObj = 0;

int preLB = 0;

boost::timer::cpu_timer timer;

float roottime;
int optimality;
float objvalue;
float rootbound;
float rootUB;
float bestUB;
float bestLB;

int sepCounter;

double preIncuObj = INT_MAX;

IloNumArray y_start;

vector<IloNumArray> y_solutions;


// y-pcenter Callbacks / fixedCustomer
ILOUSERCUTCALLBACK4(userPCYfC, Instance, myI, IloIntVarArray, y, IloIntVar, z, int, preLB) {  
    //cout << "Error in User" << endl;
    //cout << getObjValue() << " | " << getNnodes() << endl;
    int currNode = getNnodes();
    int rootNode = 0;
    IloEnv myEnv = getEnv();
    IloNumArray y_lb(getEnv(), myI.nNodes);
    getValues(y_lb,y);
    int maxNumSep;

    double z_obj = getObjValue();
    
    
    if (currNode == preNode) {
        nodeIterations++;
        if (z_obj - preZ < epsOpt) {
            boundNotImprovedCount++;
            boundNotImprovedCountFixed++;
            if (boundNotImprovedCount == maxNoImprovements) {
                cout << "boundNotImprovedCount == maxNoImprovements" << endl;
                return;
            }
        }
    } else {
        boundNotImprovedCount = 0;
        boundNotImprovedCountFixed = 0;
        nodeIterations = 0;
        maxNumSep = maxNumSepTree;
    }
    maxNumSep = maxNumSepRoot;
    if (currNode != rootNode) {
        maxNumSep = maxNumSepTree;
        //cout << "currNode != rootNode" << endl;
    }
    if (nodeIterations > maxNumSep) {
        //cout << nodeIterations << endl;
        //cout << "Too many seperation rounds" << endl;
        return;
    }

    int LB = preLB;
    //cout << initLB << endl;;
    if (ceil(getObjValue()-epsOpt) > preLB) {
        LB = ceil(getObjValue()-epsOpt);
    }// else if (preLB > initLB) {
     //   LB = preLB;
    //}

    //int LB = ceil(getObjValue()-epsOpt);
    //cout << LB << "; " << getObjValue() << endl;
    
    preZ = z_obj;
    
    preNode = currNode;
    // Seperate over iHat
    
    vector<nodeValuePair2> violatedNodes = seperationPC(y_lb, LB, iHat);
    
    // Build cut for every customer in iHat
    for (auto k:violatedNodes) {
        IloExpr myCut(myEnv);
        double RHS;

        cutViolation cV = generateCut(k, y, y_lb, myEnv, LB);
        RHS = cV.violation;
        myCut = cV.myCut;

        if (z_obj + epsOpt < RHS) {
            //cout << "First Sep cut added" << endl;
            //cout << myCut << " | " << LB << endl;
            if (currNode == rootNode) {
                add(z >= myCut, IloCplex::CutManagement::UseCutPurge);
            } else {
                addLocal(z >= myCut);
            }
            firstCuts++;
        }
        myCut.end();
    }

    

    set<int> inonHat2;
    set_difference(bigI.begin(), bigI.end(), iHat.begin(), iHat.end(), std::inserter(inonHat2, inonHat2.begin()));
    vector<nodeValuePair2> violatedNodesNon = seperationPC(y_lb, LB, inonHat2);

    //cout << violatedNodesNon[0].violation << " ; " << violatedNodesNon[1].violation << " ; " << violatedNodesNon[2].violation << endl;
    if (boundNotImprovedCountFixed < maxNoImprovementsFixed) {
        if (violatedNodesNon.size() >= 1) {
            iHat.insert(violatedNodesNon[0].cust);
            //inonHat.erase(violatedNodesNon[0].cust);
            double RHS;
            cutViolation cV = generateCut(violatedNodes[0], y, y_lb, myEnv, LB);
            IloExpr myCut = cV.myCut;
            RHS = cV.violation;
            //cout << "Second Sep cut added" << endl;
            if (z_obj + epsOpt < RHS) {
                if (currNode == rootNode) {
                    add(z >= myCut, IloCplex::CutManagement::UseCutPurge);
                } else {
                    addLocal(z >= myCut);
                }
                secondCuts++;
            }
            myCut.end();
        }
    } else {
        boundNotImprovedCountFixed = 0;
        //cout << "Third cut" << endl;
        set<int> iBar;
        set_difference(bigI.begin(), bigI.end(), iHat.begin(), iHat.end(), std::inserter(iBar, iBar.begin()));
        //iBar = inonHat2;
        for (auto k:violatedNodesNon) {
            if (iBar.find(k.cust) != iBar.end()) {
                iHat.insert(k.cust);
                //iBar.erase(k.cust);
                cutViolation cV = generateCut(k, y, y_lb, myEnv, LB);
                IloExpr myCut = cV.myCut;
                //cout << "Third Sep cut added" << endl;
                double RHS = cV.violation;
                if (z_obj + epsOpt < RHS) {
                    if (currNode == rootNode) {
                        add(z >= myCut, IloCplex::CutManagement::UseCutPurge);
                    } else {
                        addLocal(z >= myCut);
                    }
                    thirdCuts++;
                }
                myCut.end();

                for (int j = 0; j < myI.nNodes; ++j) {
                    if (getDistance(k.cust, j) <= LB) {
                        iBar.erase(j);
                    }
                }
            }
        }

    }
    //cout << "End of User Cut: " << iHat.size() << endl;
}

ILOLAZYCONSTRAINTCALLBACK4(lazySingle2, Instance, myI, IloIntVarArray, y, IloIntVar, z, int, preLB) {
    //cout << "Error in Lazy" << endl;
    int currNode = getNnodes();
    int nNodes = myI.nNodes;
    IloEnv myEnv = getEnv();
    //int LB = ceil(getBestObjValue()-epsOpt);
    IloNumArray y_lp(getEnv(), myI.nNodes);
    getValues(y_lp,y);

    int LB = preLB;
    if (ceil(getBestObjValue()-epsOpt) > preLB) {
        LB = ceil(getBestObjValue()-epsOpt);
    }
    double myObj = 0;
    int minID = -1;

    vector<nodeValuePair> openNodes;
    for (int j = 0; j < nNodes; ++j) {
        if (y_lp[j] > 0) {
            nodeValuePair nVP;
            nVP.id = j;
            nVP.value = y_lp[j];
            openNodes.push_back(nVP);
        }
    }

    int minCust = -1;
    for (int i = 0; i < myI.nNodes; ++i) {
        int minDist = INT_MAX;
        for (auto j:openNodes) {
            int distHelper = max(getDistance(i, j.id), LB);
            if (distHelper < minDist) {
                minDist = distHelper;
                minCust = j.id;
            }
        }
        if (minDist > myObj) {
            myObj = minDist;
            minID = i; 
        }
    }

    nodeValuePair2 nVP;
    nVP.cust = minID;
    nVP.criticalDistance = myObj;

    IloExpr myCut(myEnv);
    double RHS;

    cutViolation cV = generateCut(nVP, y, y_lp, myEnv, LB);
    RHS = cV.violation;
    myCut = cV.myCut;

    if (getBestObjValue() + epsOpt < RHS) {
        add(z >= myCut);
    }

    myCut.end();

    if (iHat.find(minID) == iHat.end()) {
        iHat.insert(minID);
    }
}


// nested ypcenter Callbacks
ILOLAZYCONSTRAINTCALLBACK5(lazynPCY, Instance, myI, IloIntVarArray2, y, IloIntVarArray, z, IloNumVar, w , vector<int>, LB) {

//ILOLAZYCONSTRAINTCALLBACK4(lazynPCY, Instance, myI, IloIntVarArray2, y, IloIntVarArray, z, vector<int>, LB) {
    //cout << "Error in Lazy" << endl;
    int currNode = getNnodes();
    //cout << currNode << endl;
    int nNodes = myI.nNodes;
    IloEnv myEnv = getEnv();
    int horizon = myI.endP - myI.startP;

    IloNumArray2 y_lb(myEnv, horizon);
    for (int h = 0; h < horizon; ++h) {
        y_lb[h] = IloNumArray(myEnv, nNodes);
        getValues(y_lb[h], y[h]);
    }

    for (int h = 0; h < horizon; ++h) {

        double myObj = 0;
        int minID = -1;

        vector<nodeValuePair> openNodes;
        for (int j = 0; j < nNodes; ++j) {
            if (y_lb[h][j] > 0) {
                nodeValuePair nVP;
                nVP.id = j;
                nVP.value = y_lb[h][j];
                openNodes.push_back(nVP);
            }
        }

        int minCust = -1;
        for (int i = 0; i < myI.nNodes; ++i) {
            int minDist = INT_MAX;
            for (auto j:openNodes) {
                int distHelper = max(getDistance(i, j.id), LB[h]);
                if (distHelper < minDist) {
                    minDist = distHelper;
                    minCust = j.id;
                }
            }
            if (minDist > myObj) {
                myObj = minDist;
                minID = i; 
            }
            //cout << h << ": " << LB[h] << endl;  
        }

        nodeValuePair2 nVP;
        nVP.cust = minID;
        nVP.criticalDistance = myObj;

        IloExpr myCut(myEnv);
        double RHS;

        cutViolation cV = generateNCut(nVP, y, y_lb, myEnv, LB, h);
        RHS = cV.violation;
        myCut = cV.myCut;

        if (getValue(z[h]) + epsOpt < RHS) {
            add(z[h] >= myCut);
        }

        myCut.end();

        if (iHat.find(minID) == iHat.end()) {
            iHat.insert(minID);
        }

    }

    /*
    IloNumArray z_sol(myEnv, horizon);
    double UB_test = getIncumbentObjValue();
    //double UB_test = getValue(w);
    cout << UB_test << endl;
    if (UB_test <= 5) {
        getValues(z_sol, z);
        vector<double> new_regret;
        vector<int> z_new;
        for (int h = 0; h < horizon; ++h) {
            int z_help = floor((1+UB_test)*LB[h]);
            z_new.push_back(z_help);
            double reg_help = (z_help/(double) LB[h])-1;
            new_regret.push_back(reg_help);
            cout << z_help << " | " << reg_help << " | " << LB[h] << endl;
        }
        sort(new_regret.begin(), new_regret.end(), greater<double>());
        if (UB_test - epsOpt > new_regret[0]) {
            for (int h = 0; h < horizon; ++h) {
                cout << z_sol[h] << "; " << z_new[h] << endl;
                add(z[h] <= z_new[h]);
            }
            cout << UB_test << "; " << new_regret[0] << endl;
            add(w <= new_regret[0]);
        }
    }
    else {
        cout << "UB_test: " << UB_test << endl;
    }*/
    

}


//ILOUSERCUTCALLBACK4(userPCYfCNested, Instance, myI, IloIntVarArray2, y, IloIntVarArray, z, vector<int>, LB) {  
ILOUSERCUTCALLBACK5(userPCYfCNested, Instance, myI, IloIntVarArray2, y, IloIntVarArray, z, IloNumVar, w, vector<int>, LB) {  
    //cout << "Error in User" << endl;
    //cout << getObjValue() << " | " << getNnodes() << endl;
    int currNode = getNnodes();
    int rootNode = 0;
    int horizon = LB.size();
    IloEnv myEnv = getEnv();
    IloNumArray2 y_lb(getEnv(), horizon);

    for (int h = 0; h < horizon; ++h) {
        y_lb[h] = IloNumArray(getEnv(), myI.nNodes);
        getValues(y_lb[h],y[h]);
    }

    IloNumArray z_lb(getEnv(), horizon);
    getValues(z_lb,z);

    int maxNumSep;
    
    double obj = getObjValue();

    if (currNode == preNode) {
        nnodeIterations++;
        if (obj - preObj < epsOpt) {
            nboundNotImprovedCount++;
            nboundNotImprovedCountFixed++;
            if (nboundNotImprovedCount == nmaxNoImprovements) {
                cout << "boundNotImprovedCount == maxNoImprovements" << endl;
                return;
            }
        }
    } else {
        nboundNotImprovedCount = 0;
        nboundNotImprovedCountFixed = 0;
        nnodeIterations = 0;
        maxNumSep = nmaxNumSepTree;
    }
    maxNumSep = nmaxNumSepRoot;
    if (currNode != rootNode) {
        maxNumSep = nmaxNumSepTree;
        //cout << "currNode != rootNode" << endl;
    }
    if (nnodeIterations > maxNumSep) {
        return;
    }



    preObj = obj;
    
    preNode = currNode;
    // Seperate over iHat
    for (int h = 0; h < horizon; ++h) {
        vector<nodeValuePair2> violatedNodes = seperationNpC(y_lb, LB, h, iHat);

        for (auto k:violatedNodes) {
            IloExpr myCut(myEnv);
            double RHS;

            cutViolation cV = generateNCut(k, y, y_lb, myEnv, LB, h);
            RHS = cV.violation;
            myCut = cV.myCut;

            if (z_lb[h] + epsOpt < RHS) {
                //cout << "First Sep cut added" << endl;
                //cout << myCut << " | " << LB << endl;
                if (currNode == rootNode) {
                    add(z[h] >= myCut, IloCplex::CutManagement::UseCutPurge);
                } else {
                    addLocal(z[h] >= myCut);
                }
                firstCuts++;
            }
            myCut.end();
        }

        set<int> inonHat2;
        set_difference(bigI.begin(), bigI.end(), iHat.begin(), iHat.end(), std::inserter(inonHat2, inonHat2.begin()));
        vector<nodeValuePair2> violatedNodesNon = seperationNpC(y_lb, LB, h, inonHat2);

        if (nboundNotImprovedCountFixed < nmaxNoImprovementsFixed) {
            if (violatedNodesNon.size() >= 1) {
                iHat.insert(violatedNodesNon[0].cust);
                //inonHat.erase(violatedNodesNon[0].cust);
                double RHS;
                cutViolation cV = generateNCut(violatedNodes[0], y, y_lb, myEnv, LB, h);
                IloExpr myCut = cV.myCut;
                RHS = cV.violation;
                //cout << "Second Sep cut added" << endl;
                if (z_lb[h] + epsOpt < RHS) {
                    if (currNode == rootNode) {
                        add(z[h] >= myCut, IloCplex::CutManagement::UseCutPurge);
                    } else {
                        addLocal(z[h] >= myCut);
                    }
                    secondCuts++;
                }
                myCut.end();
            }
        } else {
            nboundNotImprovedCountFixed = 0;
            //cout << "Third cut" << endl;
            set<int> iBar;
            set_difference(bigI.begin(), bigI.end(), iHat.begin(), iHat.end(), std::inserter(iBar, iBar.begin()));
            //iBar = inonHat2;
            for (auto k:violatedNodesNon) {
                if (iBar.find(k.cust) != iBar.end()) {
                    iHat.insert(k.cust);
                    //iBar.erase(k.cust);
                    cutViolation cV = generateNCut(k, y, y_lb, myEnv, LB, h);
                    IloExpr myCut = cV.myCut;
                    //cout << "Third Sep cut added" << endl;
                    double RHS = cV.violation;
                    if (z_lb[h] + epsOpt < RHS) {
                        if (currNode == rootNode) {
                            add(z[h] >= myCut, IloCplex::CutManagement::UseCutPurge);
                        } else {
                            addLocal(z[h] >= myCut);
                        }
                        thirdCuts++;
                    }
                    myCut.end();

                    for (int j = 0; j < myI.nNodes; ++j) {
                        if (getDistance(k.cust, j) <= LB[h]) {
                            iBar.erase(j);
                        }
                    }
                }
            }
        }
    }

    double LB_test = getObjValue();
    //preIncuObj = UB_test;
    //int horizon = myI.endP - myI.startP;
    //double UB_test = getValue(w);
    //cout << UB_test << endl;

    vector<double> new_regret;
    vector<int> z_new;
    for (int h = 0; h < horizon; ++h) {
        int z_help = ceil((1+LB_test)*LB[h]);
        z_new.push_back(z_help);
        double reg_help = (z_help/(double) LB[h])-1;
        new_regret.push_back(reg_help);
        //cout << z_help << " | " << reg_help << " | " << LB[h] << endl;
    }
    sort(new_regret.begin(), new_regret.end(), less<double>());
    //cout << LB_test << " | " << new_regret[0] << endl;
    if (LB_test + epsOpt < new_regret[0]) {
        if (currNode == rootNode) {
            add(w >= new_regret[0], IloCplex::CutManagement::UseCutPurge);
        } else {
            addLocal(w >= new_regret[0]);
        }
    }
    

}

// heuristic callbacks
ILOHEURISTICCALLBACK5(heurCBPC, Instance, myI, IloIntVarArray, y, IloIntVar, z, int, P, int, preLB) {
    //cout << "Error in Heuristic" << endl;
    IloNumArray y_lp(getEnv(), myI.nNodes);
    getValues(y_lp,y);

    int LB = preLB;
    if (ceil(getBestObjValue()-epsOpt) > LB) {
        LB = ceil(getBestObjValue()-epsOpt);
    }
    //int LB = ceil(getBestObjValue()-epsOpt);

    // Save all y solutions as nVP in vector if y is larger than 0
    vector<nodeValuePair> mySol;
    for (int j = 0; j < myI.nNodes; ++j) {
        if (y_lp[j] > 0) {
            nodeValuePair nVP;
            nVP.id = j;
            nVP.value=y_lp[j];
            mySol.push_back(nVP);
        }
    }

    sort(mySol.begin(), mySol.end(), std::greater<nodeValuePair>());
    
    Solution heurSol;
    heurSol.obj = INT_MAX;
    heurSol.y = vector<int>(myI.nNodes, false);

    int p = 0;
    
    int maxID = -1;     // Saves the most pushing customer
    int maxVal = 0;     // Saves the value of the most pushing customer
    
    //vector<nodeValuePair> mySol2 = mySol;
    vector<int> minDistance(myI.nNodes);

    // Fill the minDist vector with the first facility
    for (int i = 0; i < myI.nNodes; ++i) {
        minDistance[i] = max(getDistance(i, mySol[0].id), LB);
        if (minDistance[i] > maxVal) {
            maxID = i;
            maxVal = minDistance[i];
        }
    }

    heurSol.y[mySol[0].id] = 1;
    p++;

    int solSize = mySol.size();
    int stopper = 0;

    while (p < P) {
        //cout << "here" << endl;
        
        for (auto j:mySol) {
            //cout << j.id << " | " << heurSol.y[j.id] << endl;
            if (p == P) {
                break;
            }
            if (heurSol.y[j.id] == 1) {
                //cout << j.id << endl;
                continue;
            }

            int distMax = max(getDistance(maxID, j.id),LB);
            
            //cout << stopper << " | " << solSize-p-1 << endl;

            if (distMax >= maxVal && stopper < solSize-p) {
                stopper++;
                continue;
            }
            maxVal = 0;
            p++;
            stopper = 0;
            heurSol.y[j.id] = 1;
            for (int i = 0; i < myI.nNodes; ++i) {
                int distHelper = max(getDistance(i, j.id),LB);
                //cout << distHelper << " | " << minDistance[i] << " | " << maxVal << endl;
                if (distHelper < minDistance[i]) {
                    minDistance[i] = distHelper;
                }
                
                if (minDistance[i] > maxVal) {
                    maxVal = minDistance[i];
                    maxID = i;
                }
                //cout << p << " " << maxVal << endl;
            }   
        }
    }

    if (maxVal < getIncumbentObjValue()) {
        IloNumArray vals(getEnv());
        IloNumVarArray vars(getEnv());
        
        int counter = 0;
        for (int j = 0; j < myI.nNodes; ++j) {
            vars.add(y[j]);
            vals.add(heurSol.y[j]);
            if (heurSol.y[j] > 0) {
                counter++;
            }
        }
        //cout << counter << endl;

        vars.add(z);
        vals.add(maxVal);

        IloNum obj = maxVal;

        setSolution(vars, vals, obj);
        cout << "Solution added with obj. value of: " << maxVal << endl;

        vals.end();
        vars.end();
    }

    /*
    vector<int> minDistance(myI.nNodes, INT_MAX);
    vector<int> minDistance2(myI.nNodes, INT_MAX);

    while (p < P) {
        //cout << p << endl;
        for (auto j:mySol) {
            if (p == P) {
                break;
            }
            //if (heurSol.y[j.id] == 1) {
            //    continue;
            //}

            

            int improve = 0;
            for (int i = 0; i < myI.nNodes; ++i) {
                int distHelper = getDistance(i, j.id);
                if (distHelper < minDistance[i]) {
                    minDistance2[i] = distHelper;
                    improve = 1;
                }
            }

            int max_elem = *std::max_element(minDistance2.begin(), minDistance2.end());
            //cout << heurSol.obj << endl;
            if (max_elem < heurSol.obj) {
                //cout << "Added" << endl;
                //minDistance = minDistance2;
                minDistance = minDistance2;
                p++;
                heurSol.y[j.id] = 1;
                heurSol.obj = max_elem;
            }
            else {
                minDistance2 = minDistance;
            }
        }

    }


    if (heurSol.obj < getIncumbentObjValue()) {
        IloNumArray vals(getEnv());
        IloNumVarArray vars(getEnv());
        
        for (int j = 0; j < myI.nNodes; ++j) {
            vars.add(y[j]);
            vals.add(heurSol.y[j]);
        }

        vars.add(z);
        vals.add(heurSol.obj);

        setSolution(vars, vals, heurSol.obj);
        //cout << "Solution added with obj. value of: " << heurSol.obj << endl;

        vals.end();
        vars.end();
    } */

}

ILOHEURISTICCALLBACK6(heurNCBPC, Instance, myI, IloIntVarArray2, y, IloIntVarArray, z, IloNumVar, w, vector<int>, P, vector<int>, LB) {
    //cout << "In Heuristic" << endl;
    int currNode = getNnodes();
    int rootNode = 0;
    int horizon = P.size();
    IloEnv myEnv = getEnv();
    IloNumArray2 y_lb(getEnv(), horizon);
    for (int h = 0; h < horizon; ++h) {
        y_lb[h] = IloNumArray(getEnv(), myI.nNodes);
        getValues(y_lb[h],y[h]);
    }
    IloNumArray z_lb(getEnv(), horizon);
    getValues(z_lb,z);

    vector<nodeValuePair> y_sum;
    for (int j = 0; j < myI.nNodes; ++j) {
        double sum = 0;
        for (int h = 0; h < horizon; ++h) {
            /*if (y_lb[h][j] > epsOpt) {
                cout << "y_lb." << h << "." << j << "=" << y_lb[h][j] << endl;
            }*/
            
            sum += y_lb[h][j];
            //sum += round(y_lb[h][j]);
        }
        if (sum > 0) {
            nodeValuePair nVP;
            nVP.id = j;
            nVP.value = sum;
            y_sum.push_back(nVP);
        }
        
    }

    sort(y_sum.begin(), y_sum.end(), std::greater<nodeValuePair>());

    /*for (auto j:y_sum) {
        cout << "y." << j.id << "; " << j.value << endl;
    }*/

    /*for (auto j:y_sum) {
        if (j.value > 0) {
            cout << "y." << j.id << " = " << j.value << endl;      
        }
    }*/

    
    SolutionNested heurSol;
    heurSol.obj = INT_MAX;
    heurSol.y = vector<vector<int> >(horizon);
    for (int h = 0; h < horizon; ++h) {
        for (int j = 0; j < myI.nNodes; ++j) {
            heurSol.y[h].push_back(0);
        }
    }

    double heurObj = 0;
    vector<int> z_sol(horizon,0);
    vector<double> w_sol(horizon,0.0000);
    for (int h = myI.endP-1; h > myI.endP-horizon-1; --h) {
        
        //cout << h << endl;
        vector<nodeValuePair> y_help(y_sum.begin(), y_sum.begin() + h);
        int k = h-myI.startP;

        //cout << "Test: " << w_sol[k] << endl; 
        //int minDist = INT_MAX;
        //cout << k << endl;
        for (auto j:y_help) {
            heurSol.y[k][j.id] = 1;
        }

        int maxDist = 0;
        for (int i = 0; i < myI.nNodes; ++i) {
            int minDist = INT_MAX;
            for (auto j:y_help) {
                int distHelper = getDistance(i, j.id);
                if (distHelper < minDist) {
                    minDist = distHelper;
                }
            }
            if (minDist > maxDist) {
                maxDist = minDist;
            }
        }
        z_sol[k] = maxDist;
        int l = horizon-1 - k;
        double helper = z_sol[k]-LB[k];
        helper = helper/LB[k];
        //cout << helper << endl;
        if (helper > heurObj) {
            heurObj = helper;
        }
        /*w_sol[k] = helper/LB[k];

        heurObj += w_sol[k];*/

        //cout << z_sol[k] << " | " << LB[k] << " | " << w_sol[k] << endl;
        k--;
    }
    //cout << heurObj << endl;

    if (heurObj <= getIncumbentObjValue()) {
        IloNumArray vals(getEnv());
        IloNumVarArray vars(getEnv());

        IloNumArray lb(getEnv());
        IloNumArray ub(getEnv());

        for (int h = 0; h < horizon; ++h) {
            //cout << "size: " << heurSol.y[h].size() << endl;
            for (int j = 0; j < myI.nNodes; ++j) {
                if (heurSol.y[h][j] > epsOpt) {
                    vars.add(y[h][j]);
                    vals.add(heurSol.y[h][j]);
                    lb.add(heurSol.y[h][j]);
                    ub.add(1);
                }
                else {
                    vars.add(y[h][j]);
                    vals.add(0);
                    lb.add(0);
                    ub.add(1);
                }
            }
        }

        for (int h = 0; h < horizon; ++h) {
            
            vars.add(z[h]);
            vals.add(z_sol[h]);
            lb.add(z_sol[h]);
            ub.add(z_sol[h]);
        }

        double objective = 0;

        //cout << getValue(w) << endl; 

        vars.add(w);
        vals.add(heurObj);



        /*for (int h = 0; h < horizon; ++h) {
            vars.add(w[h]);
            vals.add(w_sol[h]);
            lb.add(w_sol[h]);
            ub.add(w_sol[h]);
            objective += w_sol[h];
            
        }*/

        //setBounds(vars, lb, ub);
        setSolution(vars, vals, heurObj);
        /*for (int i = 0; i < vars.getSize(); ++i) {
            if (vals[i] > 0) {
                cout << vars[i] << ": " << lb[i] << ", " << ub[i] << endl;
            }
        }*/


        //solve();
        //cout << getObjValue() << endl;
        /*for (int h = 0; h < horizon; ++h) {
            for (int j = 0; j < myI.nNodes; ++j) {
                cout << "y." << h << "." << j << ": " << getValue(y[h][j]) << endl;
            }
        }*/
        
        //cout << getStatus() << endl;
        //cout << "Solution added with obj. value of: " << heurObj << endl;
    }
   // cout << "Out of Heuristic Callback" << endl;
}

ILOMIPINFOCALLBACK0(elloumiInfo) {
    if (getNnodes() == 0) {
        roottime = getCplexTime() - getStartTime();
        rootbound = getBestObjValue();
        rootUB = getIncumbentObjValue();
    }
    bestLB = getBestObjValue();
    bestUB = getIncumbentObjValue();
}

int main(int argc, char* argv[]) {

    //timer.start();
    //timer.stop();
    filename = boost::filesystem::path(params.file).stem().string();
    ProgramOptions po(argc, argv);

    // Read in the instance
    if (params.instanceformat == 1) {
        readInstance();
        if (params.startP != 5) {
            myI.startP = params.startP;
            myI.endP = myI.startP + params.endP + 1;
        }
        else {
            myI.endP = myI.startP + params.endP + 1;
        }
    } else if (params.instanceformat == 2) {
        readInstanceTSP();
        myI.startP = params.startP;
        myI.endP = params.endP;
    } else {
        cerr << "No valid instance format was provide. Pick 1 (PMED) or 2 (TSPLIB)" << endl;
    }    

    try {
        // Nesting OFF
        if (params.nesting == 0) {
            IloEnv pcEnv;
            IloModel pcModel(pcEnv, "p-center Model");
            IloCplex pcCplex(pcModel);

            pcCplex.setParam(IloCplex::Param::WorkMem, 12000);
            
            if (params.prerun == 1) {
                cout << "Starting the non-nested with XY and p=" << myI.startP << endl;
                IloIntVarArray2 x(pcEnv, myI.nNodes);
                IloIntVarArray y(pcEnv, myI.nNodes, 0, 1);
                IloIntVar z(pcEnv);
                PC(pcModel, myI, x, y, z, myI.startP);
                //pcCplex.exportModel("test.lp");
                pcCplex.solve();
                cout << "Objective value: " << pcCplex.getObjValue() << endl;
                for (int j = 0; j < myI.nNodes; ++j) {
                    if (pcCplex.getValue(y[j]) > 0.001) {
                        cout << "y." << j << " = " << pcCplex.getValue(y[j]) << endl;
                    }
                }
            } else if (params.prerun == 2) {
                IloIntVarArray y(pcEnv, myI.nNodes, 0, 1);;
                IloIntVar z(pcEnv);
                cout << "Starting the non-nested with Y and p=" << myI.startP << endl;
                PCY(pcModel, myI, y, z, myI.startP);
                initSetfC(myI.startP);
                //pcCplex.use(lazyPCY(pcEnv, myI, y, z));
                //pcCplex.use(lazyPCYfC(pcEnv, myI, y, z));
                //pcCplex.use(lazyPCYfC2(pcEnv, myI, y, z));
                
                //pcCplex.use(lazySingle(pcEnv, myI, y, z)); // This is right 
                
                pcCplex.use(lazySingle2(pcEnv, myI, y, z, initLB));
                
                //pcCplex.use(userPCY(pcEnv, myI, y, z));
                pcCplex.use(userPCYfC(pcEnv, myI, y, z, initLB));
                
                pcCplex.use(heurCBPC(pcEnv, myI, y, z, myI.startP, initLB));

                //setCPLEXParameters(pcCplex);
                
                pcCplex.solve();

                cout << "First Cuts: " << firstCuts << endl;
                cout << "Second Cuts: " << secondCuts << endl;
                cout << "Third Cuts: " << thirdCuts << endl;
                cout << "Objective value: " << pcCplex.getObjValue() << endl;
                for (int j = 0; j < myI.nNodes; ++j) {
                    if (pcCplex.getValue(y[j]) > 0.001) {
                        cout << "y." << j << " = " << pcCplex.getValue(y[j]) << endl;
                    }
                }
                //boost::timer::cpu_times times = timer.elapsed();
                cout << timer.format() << "; " << sepCounter <<  endl;
            } else if (params.prerun == 3) {
                IloIntVarArray y(pcEnv, myI.nNodes, 0, 1);
                IloIntVarArray u(pcEnv, (int) myI.distinct_distances.size(), 0, 1);
                IloIntVar R(pcEnv);
                cout << "Starting the non-nested with Elloumi and p=" << myI.startP << endl;
                PCE(pcModel, myI, u, y, R, myI.startP);
                pcCplex.exportModel("test.lp");
                pcCplex.solve();
                cout << "Objective value: " << pcCplex.getObjValue() << endl;
                for (int j = 0; j < myI.nNodes; ++j) {
                    if (pcCplex.getValue(y[j]) > 0.001) {
                        cout << "y." << j << " = " << pcCplex.getValue(y[j]) << endl;
                    }
                }
                
            } else {
                cerr << "No valid model was provided. 1=XY, 2=Y 3=Elloumi" << endl;
            }

            pcEnv.end();
        // Nesting ON
        } else {
            // Create set P           
            vector<int> P;

            IloEnv npcEnv;
            IloModel npcModel(npcEnv, "nested p-center Model");
            IloCplex npcCplex(npcModel);

            for (int i = 0; i < myI.nNodes; ++i) {
                    bigI.insert(i);
                }

            for (int p = 0; p < myI.endP - myI.startP; ++p) {
                P.push_back(myI.startP + p);
            }

            // Select pre runs:
            vector<int> LBs;
            if (params.prerun == 0) {
                for (int h = 0; h < P.size(); ++h) {
                    LBs.push_back(0);
                }
            } else if (params. prerun == 1) {

            }

            // Heuristics and in heuristics
            tuple<int, int> bounds;
            if (params.heuristic == 0) {
                //bounds = make_tuple(0, INT_MAX);
                //bounds = make_tuple(108, 140);

            } else if (params.heuristic == 1) {
                bounds = highPHeuristic();

            } else if (params.heuristic == 2) {
                bounds = lowPHeuristic();

            } else if (params.heuristic == 3) {
                bounds = highLowHeuristic();
            } else {
                cerr << "No valid heuristic was provided! 0=Off, 1=HighP, 2=LowP, 3=HighLow" << endl;
            }

            if (params.nesting == 1) {
                cout << "Starting the non-nested with XY and P={" << myI.startP << ", " << myI.endP << "}" << endl;
                for (int p = 0; p < P.size(); ++p) {
                    cout << P[p] << endl;
                }
                IloIntVarArray3 x(npcEnv, P.size());
                IloIntVarArray2 y(npcEnv, P.size());
                IloIntVarArray z(npcEnv, P.size());
                nPC(npcModel, myI, x, y, z, P, bounds);
                npcCplex.setParam(IloCplex::Param::Threads, 1);
                //pcCplex.exportModel("test.lp");
                npcCplex.solve();
                cout << "Objective value: " << npcCplex.getObjValue() << endl;
                for (int h = 0; h < P.size(); ++h) {
                    for (int j = 0; j < myI.nNodes; ++j) {
                        if (npcCplex.getValue(y[h][j]) > 0.001) {
                            cout << "y." << h << "." << j << " = " << npcCplex.getValue(y[h][j]) << endl;
                        }
                    }
                }

            } else if (params.nesting == 2) {
                cout << "Starting the nested with Y and P={" << myI.startP << ", " << myI.endP-1 << "}" << endl;
                IloIntVarArray2 y(npcEnv, P.size());
                IloIntVarArray z(npcEnv, P.size());
                //IloNumVarArray w(npcEnv, P.size());
                IloNumVar w(npcEnv);

                /*for (int i = 0; i < myI.nNodes; ++i) {
                    for (int j = 0; j < myI.nNodes; ++j) {
                        cout << getDistance(i,j) << ", " ;
                    } cout << endl;
                }*/

                tuple<int, int> bounds;
                vector<int> LBs;
                preLB = initLB;
                /*for (int h = 0; h < P.size(); ++h) {
                    IloModel preRun(npcEnv, "Prerun");
                    cout << "preLB " << preLB << endl;
                    int obj = yPreRun(preRun, P[h]);
                    LBs.push_back(obj);
                    //LBs.insert(LBs.begin(), obj);
                    preLB = obj;
                    preRun.end();
                    cout << LBs[h] << endl;
                }*/
                
                for (int h = P.size()-1; h >= 0; --h) {
                    IloModel preRun(npcEnv, "Prerun");
                    int obj = yPreRun2(preRun, P[h], preLB);
                    LBs.insert(LBs.begin(), obj);
                    preLB = obj;
                    preRun.end();
                    //cout << LBs[h] << endl;
                    if (h == P.size()-1) {

                    }
                }

                 //nPCY(npcModel, myI, y, z, P, bounds);
                nPCYRegretMinMax(npcModel, myI, y, z, w, P, LBs);
                initSetfC(myI.endP-1);

                //setCPLEXParameters(npcCplex);

                //cout << "MIP Start 2 is started" << endl;
                MIPStart2(npcEnv, y, z, npcCplex);
                //cout << "MIP Start 2 is finished" << endl;

                MIPStart(npcEnv, y, z, npcCplex);

                //MIPStart3(npcEnv, y, z, npcCplex, y_solutions);
                
                
                /*
                set<int> helperSet;
                vector<nodeValuePair> openFacilities;
                for (int h = 0; h < P.size(); ++h) {
                    for (int j = 0; j < myI.nNodes; ++j) {
                        if (y_solutions[h][j] > 0.9) {
                            cout << "y." << h << "." << j << ": " << y_solutions[h][j] << endl;
                            if (helperSet.find(j) == helperSet.end()) {
                                helperSet.insert(j);
                                nodeValuePair nVP;
                                nVP.id = j;
                                nVP.distance = 1;
                                openFacilities.push_back(nVP);
                            }
                            else {
                                for (int k = 0; k < openFacilities.size(); ++k) {
                                    //cout << openFacilities[k].id << "; " << j << endl;
                                    if (openFacilities[k].id == j) {
                                        openFacilities[k].distance ++;
                                        //cout << openFacilities[k].distance << endl;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }

                sort(openFacilities.begin(), openFacilities.end(), std::greater<nodeValuePair>());
                for (auto j:openFacilities) {
                    if (j.distance > 1) {
                        cout << j.distance << endl;
                        for (int h = 0; h < P.size(); ++h) {
                            npcCplex.setPriority(y[h][j.id], 100);
                            cout << "Priority y." << h << "." << j.id << endl;
                        }
                    }
                    else {
                        for (int h = 0; h < P.size(); ++h) {
                            npcCplex.setPriority(y[h][j.id], 5);
                            cout << "Priority y." << h << "." << j.id << endl;
                        }
                    }
                }*/

                for (int h = 0; h < P.size(); ++h) {
                    npcCplex.setPriority(z[h], 50);
                }
                


                npcCplex.use(elloumiInfo(npcEnv));
               
                
                npcCplex.setParam(IloCplex::Param::Threads, 1);
                npcCplex.use(lazynPCY(npcEnv, myI, y, z, w, LBs));
                //npcCplex.use(lazynPCY(npcEnv, myI, y, z, LBs));
                //npcCplex.use(usernPCY(npcEnv, myI, y, z, LBs));
                //npcCplex.use(userPCYfCNested(npcEnv, myI, y, z, LBs));
                npcCplex.use(userPCYfCNested(npcEnv, myI, y, z, w, LBs));
                npcCplex.exportModel("test.lp");

                npcCplex.use(heurNCBPC(npcEnv, myI, y, z, w, P, LBs));

                npcCplex.solve();
                objvalue = npcCplex.getObjValue();

                
                cout << "Objective value: " << npcCplex.getObjValue() << endl;

                for (int h = 0; h < P.size(); ++h) {
                    for (int j = 0; j < myI.nNodes; ++j) {
                        if (npcCplex.getValue(y[h][j]) > 0.001) {
                            cout << "y." << h << "." << j << " = " << npcCplex.getValue(y[h][j]) << endl;
                        }
                    }
                }

                for (int h = 0; h < P.size(); ++h) {
                    cout << "z." << h << " = " << npcCplex.getValue(z[h]) << endl;
                }

                cout << "Root Lazy Cuts: " << rootLazyCuts << endl;
                cout << "Root User Cuts: " << rootUserCuts << endl;
                cout << "Lazy Cuts: " << lazyCuts << endl;
                cout << "User Cuts: " << userCuts << endl;

                //cplextime = masterCplex.getCplexTime() - cplextime;
                printSTATs(npcEnv, myI, P);

            } else if (params.nesting == 3) {
                cout << "Starting the non-nested with XY and P={" << myI.startP << ", " << myI.endP << "}" << endl;
                for (int p = 0; p < P.size(); ++p) {
                    cout << P[p] << endl;
                }
                IloIntVarArray2 u(npcEnv, P.size());
                IloIntVarArray2 y(npcEnv, P.size());
                IloIntVarArray R(npcEnv, P.size());

                nPCE(npcModel, myI, u, y, R, P, bounds);
                npcCplex.setParam(IloCplex::Param::Threads, 1);
                //pcCplex.exportModel("test.lp");
                npcCplex.solve();
                cout << "Objective value: " << npcCplex.getObjValue() << endl;

                for (int h = 0; h < P.size(); ++h) {
                    for (int j = 0; j < myI.nNodes; ++j) {
                        if (npcCplex.getValue(y[h][j]) > 0.001) {
                            cout << "y." << h << "." << j << " = " << npcCplex.getValue(y[h][j]) << endl;
                        }
                    }
                }
            } else {
                cerr << "No valid nesting mode provided! 0=Off, 1=XY, 2=Y, 3=Elloumi" << endl;
            }
        }
    }
    catch (const IloException& e) {
        cerr << "Exception caught: " << e << endl;
    }
    catch (...) {
        cerr << "Unknown exception caught!" << endl;
    }
    

}

void readInstance() {
    myI.format = 1;
    ifstream infile(params.file);
    std::string line;
    istringstream iss(line);
    
    // Get first line of the file
    getline(infile, line);

    // Convert line into inputstream and split it into number of nodes, #edges
    iss.str(line);
    iss >> myI.nNodes;
    int nEdges = 0;
    iss >> nEdges;
    iss >> myI.startP;
    iss.clear();

    // Initialize the distance matrix with values of 50000, for the algorithm
    myI.distances = vector<vector<int>>(myI.nNodes, vector<int>(myI.nNodes, 50000));

    // Work through every Edge in the file -> read line in to i, j and helper and set the distances we already know
    for (int k = 0; k < nEdges; ++k)
    {
        getline(infile, line);
        iss.clear();
        iss.str(line);
        int i, j, helper;
        iss >> i >> j >> helper;
        --i;
        --j;
        myI.distances[i][j] = helper;
        myI.distances[j][i] = helper;
        myI.distances[i][i] = 0;
    }

    // Apply Floyd-Warshall-Algorithm
    for (int k = 0; k < myI.nNodes; k++) {
        for (int i = 0; i < myI.nNodes; i++) {
            for (int j = 0; j < myI.nNodes; j++) {
                if (myI.distances[i][j] > myI.distances[i][k] + myI.distances[k][j]) {
                    myI.distances[i][j] = myI.distances[i][k] + myI.distances[k][j];

                }
            }
        }
    }
    infile.close();

    for (int i = 0; i < myI.nNodes; i++) {
        for (int j = 0; j < myI.nNodes; j++) {
            myI.distinct_distances.insert(getDistance(i, j));
        }
    }

    // Create the Si sets for the CP1 model by Elloumi
    myI.bigS = vector<map<int, int>>(myI.nNodes);
    map<int, int> distinctMap;
        set<int>::iterator it_k = myI.distinct_distances.begin();
        for (int k = 0; k < myI.distinct_distances.size(); ++k) {
            distinctMap.insert(std::pair<int, int>(*it_k, k));
            advance(it_k, 1);
        }
        //cout << distinctMap.size() << endl;

    for (int i = 0; i < myI.nNodes; ++i) {
        for (int j = 0; j < myI.nNodes; j++) {
            if (myI.distinct_distances.find(getDistance(i, j)) != myI.distinct_distances.end()) {
                int distHelper = getDistance(i, j);
                int secHelper = distinctMap[distHelper];
                //cout << secHelper << "&" << getDistance(i, j) << endl;
                myI.bigS[i].insert(std::pair<int, int>(distHelper, secHelper));
            }
        }
        set<int>::iterator it = myI.distinct_distances.end();
        advance(it, -1);
        myI.bigS[i].insert(std::pair<int, int>(*it, myI.distinct_distances.size()-1));
        //cout << *myI.distinct_distances.end() << " & " << myI.distinct_distances.size()-1 << endl;
        //cout  << myI.bigS[i].size() << endl;
    }

}

void readInstanceTSP() {
    myI.format = 2;
    ifstream infile(params.file);
    std::string line;
    istringstream iss(line);

    while(std::getline(infile, line)) {
        std::size_t found = line.find("NODE");
        if (found == 0) {
            break;
        }
    }
    

    myI.nNodes = 0;
    while (std::getline(infile, line)){
        std::size_t found = line.find("EOF");
        if (found!=std::string::npos) {
            continue;
        }

        iss.clear();
        iss.str(line);

        ++myI.nNodes;
        double h, v1, v2;
        iss >> h >> v1 >> v2;

        myI.coords.push_back(make_pair(v1,v2));
    }

    infile.close();
    
    if (params.prerun == 3 || params.nesting == 3) {

        for (int i = 0; i < myI.nNodes; i++) {
            for (int j = 0; j < myI.nNodes; j++) {
                myI.distinct_distances.insert(getDistance(i,j));
            }
        }

        
            
        // Create the Si sets for the CP1 model by Elloumi
        myI.bigS = vector<map<int, int>>(myI.nNodes);
        map<int, int> distinctMap;
            set<int>::iterator it_k = myI.distinct_distances.begin();
            for (int k = 0; k < myI.distinct_distances.size(); ++k) {
                distinctMap.insert(std::pair<int, int>(*it_k, k));
                advance(it_k, 1);
            }

        for (int i = 0; i < myI.nNodes; ++i) {
            for (int j = 0; j < myI.nNodes; j++) {
                if (myI.distinct_distances.find(getDistance(i, j)) != myI.distinct_distances.end()) {
                    int distHelper = getDistance(i, j);
                    int secHelper = distinctMap[distHelper];
                    myI.bigS[i].insert(std::pair<int, int>(distHelper, secHelper));
                }
            }
            set<int>::iterator it = myI.distinct_distances.end();
            advance(it, -1);
            myI.bigS[i].insert(std::pair<int, int>(*it, myI.distinct_distances.size()-1));
        }
    }

    
}

inline int getDistance(int i, int j) {
    if (params.instanceformat == 2) {
        pair<double, double> p1 = myI.coords[i];
        pair<double, double> p2 = myI.coords[j];
        //double distance = round(sqrt(pow(p1.first - p2.first, 2.0) + pow(p1.second - p2.second, 2.0))); 
        double distance = (int)(sqrt(pow(p1.first - p2.first, 2.0) + pow(p1.second - p2.second, 2.0))+0.5); 
        //cout << distance << endl;
        return distance;
    }
    else {
        return myI.distances[i][j];
    }
}

void PC(IloModel myModel, Instance myI, IloIntVarArray2 x, IloIntVarArray y, IloIntVar z, int p) {
    IloInt i, j, nNodes;
    IloEnv env = myModel.getEnv();

    nNodes = myI.nNodes;
    char varName[100];

    //x = IloIntVarArray2(env, nNodes);
    //y = IloIntVarArray(env, nNodes, 0, 1);
    //z = IloIntVar(env);

    // Create variables x(i,j)
    for (i = 0; i < nNodes; ++i) {
        x[i] = IloIntVarArray(env, nNodes, 0, 1);
        for (j = 0; j < nNodes; ++j) {
            sprintf(varName, "x.%d.%d", (int) i, (int) j);
            x[i][j].setName(varName);
        }
        myModel.add(x[i]);
    }
    
    // Create variables y(j)
    for (j = 0; j < nNodes; ++j) {
        sprintf(varName, "y.%d", (int) j);
        y[j].setName(varName);
    }
    myModel.add(y);

    // Create variables z
    z.setName("z");
    myModel.add(z);
   
    // Objective function
    IloExpr obj(env);
    obj += z;
    myModel.add(IloMinimize(env, obj));
    obj.end();

    IloExpr expr(env);
    for (j = 0; j < nNodes; ++j) {
        expr += y[j];
    }
    myModel.add(expr == p);
    expr.end();

    // Assignment constraint
    for (i = 0; i < nNodes; ++i) {
        IloExpr expr(env);
        for (j = 0; j < nNodes; ++j) {
            expr += x[i][j];
        }
        myModel.add(expr == 1);
        expr.end();
    }

    // Open facility constraints
    for (i = 0; i < nNodes; ++i) {
        for (j = 0; j < nNodes; ++j) {
            myModel.add(x[i][j] <= y[j]);
        }
    }

    // z push constraint
    for (i = 0; i < nNodes; ++i) {
        IloExpr expr(env);
        for (j = 0; j < nNodes; ++j) {
            expr += getDistance(i,j)*x[i][j];
        }
        myModel.add(z >= expr);
        expr.end();
    }

}

void PCE(IloModel myModel, Instance myI, IloIntVarArray u, IloIntVarArray y, IloIntVar R, int p) {
    IloInt i, j, k, K, nNodes;
    IloEnv env = myModel.getEnv();
    
    nNodes = myI.nNodes;
    K = myI.distinct_distances.size();

    char varName[100];

    // Variable definition
    //u = IloIntVarArray(env, K, 0, 1);
    //y = IloIntVarArray(env, nNodes, 0, 1);
    //R = IloIntVar(env, 0, IloIntMax);

    for (k = 0; k < K; ++k) {
        sprintf(varName, "u.%d", (int) k);
        u[k].setName(varName);
    }
    myModel.add(u);

    for (j = 0; j < nNodes; ++j) {
        sprintf(varName, "y.%d", (int) j);
        y[j].setName(varName);
    }
    myModel.add(y);

    R.setName("R");
    myModel.add(R);

    // Adding objective function
    IloExpr obj(env);
    obj += R;
    myModel.add(IloMinimize(env, obj));
    obj.end();

    // Adding facility constraints
    IloExpr expr(env);
    for (j = 0; j < nNodes; ++j) {
        expr += y[j];
    }
    myModel.add(expr == p);
    expr.end();

    // Adding R push constraint
    IloExpr cut(env);
    auto it_k = myI.distinct_distances.begin();
    auto it_k0 = myI.distinct_distances.begin();
    cut += *it_k0;
    for (k = 1; k < K; ++k) {
        it_k++;
        cut += (*it_k - *it_k0)*u[k];
        it_k0++;
    }
    myModel.add(cut <= R);
    cut.end();

    // Adding U nested constraint
    for (k = 0; k < K-1; ++k) {
        myModel.add(u[k]>=u[k+1]);
    }

    // Adding u constraint
    for (i = 0; i< nNodes; ++i) {
        for (auto dK:myI.bigS[i]) {
            IloExpr expr(env);
            expr += u[dK.second];
            for (j = 0; j < nNodes; ++j) {
                if (getDistance(i, j) < dK.first) {
                    expr += y[j];
                }
            }
            myModel.add(expr >= 1);
            expr.end();
        }
    }
}

void PCY(IloModel myModel, Instance myI, IloIntVarArray y, IloIntVar z, int p) {
    IloInt i, j, nNodes;
    nNodes = myI.nNodes;
    IloEnv env = myModel.getEnv();

    char varName[100];

    //y = IloIntVarArray(env, nNodes, 0, 1);
    //z = IloIntVar(env, 0, IloIntMax);

    for (j = 0; j < nNodes; ++j) {
        y[j] = IloIntVar(env, 0, 1);
        sprintf(varName, "y.%d", (int) j);
        y[j].setName(varName);
    }
    myModel.add(y);

    z.setName("z");
    myModel.add(z);

    // Objective function
    IloExpr obj(env);
    obj += z;
    myModel.add(IloMinimize(env, obj));
    obj.end();

    IloExpr expr(env);
    for (j = 0; j < nNodes; ++j) {
        expr += y[j];
    }
    myModel.add(expr == p);
    expr.end();

}

void nPC(IloModel myModel, Instance myI, IloIntVarArray3 x, IloIntVarArray2 y, IloIntVarArray z,  vector<int> P, tuple<int, int> bounds){
    IloInt i, j, h, nNodes, horizon;
    IloEnv env = myModel.getEnv();

    int LB, UB;
    tie(LB, UB) = bounds;

    nNodes = myI.nNodes;
    horizon = P.size();
  
    char varName[100];

    // Declare variabels
    for (h = 0; h < horizon; ++h) {
        x[h] = IloIntVarArray2(env, nNodes);
        for (i = 0; i < nNodes; ++i) {
            x[h][i] = IloIntVarArray(env, nNodes, 0, 1);
            for (j = 0; j < nNodes; ++j) {
                sprintf(varName, "x.%d.%d.%d", (int) h, (int) i, (int) j);
                x[h][i][j].setName(varName);
            }
            myModel.add(x[h][i]);
        }
    }

    for (h = 0; h < horizon; ++h) {
        y[h] = IloIntVarArray(env, nNodes, 0, 1);
        for (j = 0; j < nNodes; ++j) {
            sprintf(varName, "y.%d.%d", (int) h, (int) j);
            y[h][j].setName(varName);
        }
        myModel.add(y[h]);
    }

    for (h = 0; h < horizon; ++h) {
        z[h] = IloIntVar(env);
        sprintf(varName, "z.%d", (int) h);
        z[h].setName(varName);
         
    }
    myModel.add(z);

    // Objective function
    IloExpr obj(env);
    for (h = 0; h < horizon; ++h) {
        obj += z[h];
    }
    myModel.add(IloMinimize(env, obj));
    obj.end();

    // # Facility constraints
    for (h = 0; h < horizon; ++h) {
        IloExpr expr(env);
        for (j = 0; j < nNodes; ++j) {
            expr += y[h][j];
        }
        myModel.add(expr == P[h]);
        expr.end();
    }

    // Assignment constraints
    for (h = 0; h < horizon; ++h) {
        for (i = 0; i < nNodes; ++i) {
            IloExpr expr(env);
            for (j = 0; j < nNodes; ++j) {
                expr += x[h][i][j];
            }
            myModel.add(expr == 1);
            expr.end();
        }
    }

    // obj. push constraint
    for (h = 0; h < horizon; ++h) {
        for (i = 0; i < nNodes; ++i) {
            IloExpr expr(env);
            for (j = 0; j < nNodes; ++j) {
                expr += max((int) getDistance(i,j), LB)*x[h][i][j];
            }
            myModel.add(expr <= z[h]);
            expr.end();
        }
    }

    // Open facility
    for (h = 0; h < horizon; ++h) {
        for (i = 0; i < nNodes; ++i) {
            for (j = 0; j < nNodes; ++j) {
                myModel.add(x[h][i][j] <= y[h][j]);
            }
        }
    }

    // Nesting constraint
    for (h = 0; h < horizon-1; ++h) {
        for (j = 0; j < nNodes; ++j) {
            myModel.add(y[h][j] <= y[h+1][j]);
        }
    }
}

void nPCE(IloModel myModel, Instance myI, IloIntVarArray2 u, IloIntVarArray2 y, IloIntVarArray R, vector<int> P, tuple<int, int> bounds) {
    IloInt i, j, k , h, nNodes, horizon, K;
    IloEnv env = myModel.getEnv();
    nNodes = myI.nNodes;
    horizon = P.size();
    K = myI.distinct_distances.size();

    int LB, UB;
    tie(LB, UB) = bounds;

    char varName[100];

    for (h = 0; h < horizon; ++h) {
        u[h] = IloIntVarArray(env, K, 0, 1);
        for (k = 0; k < K; ++k) {
            sprintf(varName, "u.%d.%d", (int) h, (int) k);
            u[h][k].setName(varName);
        }
        myModel.add(u[h]);
    }

    for (h = 0; h < horizon; ++h) {
        y[h] = IloIntVarArray(env, nNodes, 0, 1);
        for (j = 0; j < nNodes; ++j) {
            sprintf(varName, "y.%d.%d", (int) h, (int) j);
            y[h][j].setName(varName);
        }
        myModel.add(y[h]);
    }

    cout << "HERE" << endl;
    for (h = 0; h < horizon; ++h) {
        R[h] = IloIntVar(env);
        sprintf(varName, "r.%d", (int) h);
        R[h].setName(varName);
    }
    myModel.add(R);


    // Objective function
    IloExpr obj(env);
    for (h = 0; h < horizon; ++h) {
        obj += R[h];
    }
    myModel.add(IloMinimize(env, obj));
    obj.end();

    // Facility constraints
    for (h = 0; h < horizon; ++h) {
        IloExpr expr(env);
        for (j = 0; j < nNodes; ++j) expr += y[h][j];
        myModel.add(expr == P[h]);
        expr.end();
    }


    // R push constraint
    for (h = 0; h < horizon; ++h) {
        IloExpr expr(env);
        set<int>::iterator it_k = myI.distinct_distances.begin();
        set<int>::iterator it_k0 = myI.distinct_distances.begin();
        expr += *it_k0;
        for (k = 1; k < K; ++k) {
            advance(it_k, 1);
            expr += (*it_k - *it_k0)*u[h][k];
            advance(it_k0, 1);
        }
        myModel.add(expr <= R[h]);
        expr.end();
    }

    // nesting constraint
    for (h = 1; h < horizon; ++h) {
        for (j = 0; j < nNodes; ++j) {
            myModel.add(y[h-1][j] <= y[h][j]);
        }
    }

    // k nested constraint (Eq.4 Elloumi 2018)
    for (h = 0; h < horizon; ++h) {
        for (k = 0; k < K-1; ++k) {
            myModel.add(u[h][k] >= u[h][k+1]);
        }
    }
    
    // u constraint
    for (h = 0; h < horizon; ++h) {
        for (i = 0; i < nNodes; ++i) {
            for (auto dK:myI.bigS[i]) {
                IloExpr expr(env);
                expr += u[h][dK.second];
                //cout << dK.second << endl;
                for (j = 0; j < nNodes; j++) {
                    if (getDistance(i, j) < dK.first) {
                        expr += y[h][j];
                    }
                }
                myModel.add(expr >= 1);
                //cout << expr << endl;
                expr.end();
            }
        }
    }
}

void nPCY(IloModel myModel, Instance myI, IloIntVarArray2 y, IloIntVarArray z, vector<int> P, tuple<int, int> bounds) {
    IloInt i, j, h, nNodes, horizon;
    IloEnv env = myModel.getEnv();

    int LB, UB;
    tie(LB, UB) = bounds;

    nNodes = myI.nNodes;
    horizon = P.size();
  
    char varName[100];

    // Declare variabels
    for (h = 0; h < horizon; ++h) {
        y[h] = IloIntVarArray(env, nNodes, 0, 1);
        for (j = 0; j < nNodes; ++j) {
            sprintf(varName, "y.%d.%d", (int) h, (int) j);
            y[h][j].setName(varName);
        }
        myModel.add(y[h]);
    }

    for (h = 0; h < horizon; ++h) {
        z[h] = IloIntVar(env);
        sprintf(varName, "z.%d", (int) h);
        z[h].setName(varName);
         
    }
    myModel.add(z);

    // Objective function
    IloExpr obj(env);
    for (h = 0; h < horizon; ++h) {
        obj += z[h];
    }
    myModel.add(IloMinimize(env, obj));
    obj.end();

    // # of facility constraints
    for (h = 0; h < horizon; ++h) {
        IloExpr expr(env);
        for (j = 0; j < nNodes; ++j) {
            expr += y[h][j];
        }
        myModel.add(expr == P[h]);
        expr.end();
    }

    // nesting constraint
    for (h = 0; h < horizon-1; ++h) {
        for (j = 0; j < nNodes; ++j) {
            myModel.add(y[h][j] <= y[h+1][j]);
        }
    }
}

void nPCYRegret(IloModel myModel, Instance myI, IloIntVarArray2 y, IloIntVarArray z, IloNumVarArray w, vector<int> P, vector<int> opt) {
    IloInt i, j, h, nNodes, horizon;
    IloEnv env = myModel.getEnv();

    nNodes = myI.nNodes;
    horizon = P.size();
  
    char varName[100];

    // Declare variabels
    for (h = 0; h < horizon; ++h) {
        y[h] = IloIntVarArray(env, nNodes, 0, 1);
        for (j = 0; j < nNodes; ++j) {
            sprintf(varName, "y.%d.%d", (int) h, (int) j);
            y[h][j].setName(varName);
        }
        myModel.add(y[h]);
    }

    for (h = 0; h < horizon; ++h) {
        z[h] = IloIntVar(env);
        sprintf(varName, "z.%d", (int) h);
        z[h].setName(varName);
         
    }
    myModel.add(z);

    for (h = 0; h < horizon; ++h) {
        w[h] = IloNumVar(env);
        sprintf(varName, "w.%d", (int) h);
        w[h].setName(varName);
    }
    myModel.add(w);

    // Objective function
    IloExpr obj(env);
    for (h = 0; h < horizon; ++h) {
        obj += w[h];
    }
    myModel.add(IloMinimize(env, obj));
    obj.end();

    // # of facility constraints
    for (h = 0; h < horizon; ++h) {
        IloExpr expr(env);
        for (j = 0; j < nNodes; ++j) {
            expr += y[h][j];
        }
        myModel.add(expr == P[h]);
        expr.end();
    }

    // nesting constraint
    for (h = 0; h < horizon-1; ++h) {
        for (j = 0; j < nNodes; ++j) {
            myModel.add(y[h][j] <= y[h+1][j]);
        }
    }

    // regret constraint
    for (h = 0; h < horizon; ++h) {
        myModel.add(w[h] >= ((z[h]-opt[h])/opt[h]));
    }
}

void nPCYRegretMinMax(IloModel myModel, Instance myI, IloIntVarArray2 y, IloIntVarArray z, IloNumVar w, vector<int> P, vector<int> opt) {
    IloInt i, j, h, nNodes, horizon;
    IloEnv env = myModel.getEnv();

    nNodes = myI.nNodes;
    horizon = P.size();
  
    char varName[100];

    // Declare variabels
    for (h = 0; h < horizon; ++h) {
        y[h] = IloIntVarArray(env, nNodes, 0, 1);
        for (j = 0; j < nNodes; ++j) {
            sprintf(varName, "y.%d.%d", (int) h, (int) j);
            y[h][j].setName(varName);
        }
        myModel.add(y[h]);
    }

    for (h = 0; h < horizon; ++h) {
        z[h] = IloIntVar(env);
        sprintf(varName, "z.%d", (int) h);
        z[h].setName(varName);
        z[h].setBounds(opt[h], INT_MAX);
        cout << opt[h] << endl;
         
    }
    myModel.add(z);

    /*for (h = 0; h < horizon; ++h) {
        w[h] = IloNumVar(env);
        sprintf(varName, "w.%d", (int) h);
        w[h].setName(varName);
    }
    myModel.add(w);*/

    double upperBound = ((opt[0] - opt[horizon-1])/ (double) opt[horizon-1]);
    cout << "upperBound =" << upperBound << endl;
    w.setName("w");
    myModel.add(w);
    w.setBounds(0, upperBound);


    // Objective function
    IloExpr obj(env);
    /*for (h = 0; h < horizon; ++h) {
        obj += w[h];
    }*/
    obj += w;
    myModel.add(IloMinimize(env, obj));
    obj.end();

    // # of facility constraints
    for (h = 0; h < horizon; ++h) {
        IloExpr expr(env);
        for (j = 0; j < nNodes; ++j) {
            expr += y[h][j];
        }
        myModel.add(expr == P[h]);
        expr.end();
    }

    // nesting constraint
    for (h = 0; h < horizon-1; ++h) {
        for (j = 0; j < nNodes; ++j) {
            myModel.add(y[h][j] <= y[h+1][j]);
        }
    }

    // regret constraint
    for (h = 0; h < horizon; ++h) {
        myModel.add(w >= ((z[h]-opt[h])/opt[h]));
    }
}

tuple<int, int> highPHeuristic() {

}

tuple<int, int> lowPHeuristic() {

}

tuple<int, int> highLowHeuristic() {

}

int xyPreRun(IloModel myModel, int p) {
    cout << "Starting the non-nested with XY and p=" << p << endl;
    IloEnv myEnv = myModel.getEnv();
    IloCplex myCplex(myModel);
    IloIntVarArray2 x(myEnv, myI.nNodes);
    IloIntVarArray y(myEnv, myI.nNodes, 0, 1);
    IloIntVar z(myEnv);
    PC(myModel, myI, x, y, z, p);
    myCplex.solve();
    cout << "Objective value: " << myCplex.getObjValue() << endl;

    return myCplex.getObjValue();
}

int yPreRun(IloModel myModel, int p) {
    cout << "Starting the non-nested with Y and p=" << p << endl;
    IloEnv myEnv = myModel.getEnv();
    IloCplex myCplex(myModel);
    IloIntVarArray y(myEnv, myI.nNodes, 0, 1);;
    IloIntVar z(myEnv);
    //setCPLEXParameters(myCplex);
    PCY(myModel, myI, y, z, p);
    initSetfC(p);
    myCplex.use(lazySingle2(myEnv, myI, y, z, initLB));
    myCplex.use(userPCYfC(myEnv, myI, y, z, initLB));
    myCplex.use(heurCBPC(myEnv, myI, y, z, p, initLB));
    myCplex.solve();
    cout << "Objective value: " << myCplex.getObjValue() << endl;
    return myCplex.getObjValue();
}

int yPreRun2(IloModel myModel, int p, int LB) {
    cout << "Starting the non-nested with Y and p=" << p << " and an initial LB=" << LB << endl;
    IloEnv myEnv = myModel.getEnv();
    IloCplex myCplex(myModel);
    IloIntVarArray y(myEnv, myI.nNodes, 0, 1);;
    IloIntVar z(myEnv);
    //setCPLEXParameters(myCplex);
    PCY(myModel, myI, y, z, p);
    initSetfC(p);
    myCplex.use(lazySingle2(myEnv, myI, y, z, LB));
    myCplex.use(userPCYfC(myEnv, myI, y, z, LB));
    myCplex.use(heurCBPC(myEnv, myI, y, z, p, LB));
    myCplex.solve();
    cout << "Objective value: " << myCplex.getObjValue() << endl;
    for (int i = 0; i < myI.nNodes; ++i) {
        if (myCplex.getValue(y[i]) > epsOpt) {
            cout << "y." << i << ": " << myCplex.getValue(y[i]) << endl;
        }
    }

    IloNumArray y_help(myEnv);
    myCplex.getValues(y_help, y);
    y_solutions.push_back(y_help);

    if (p == myI.endP-1) {
        y_start = IloNumArray(myEnv, myI.nNodes);
        myCplex.getValues(y_start, y);
        cout << "p for the MIP Start Equals: " << p << endl; 
    }
    

    return myCplex.getObjValue();
}

void MIPStart(IloEnv env, IloIntVarArray2 y, IloIntVarArray z, IloCplex cplex) {
    // Find the facility with the least impact on the objective.
    // Then add the y values to vals and vars and then add it as MIP Start

    

    vector<nodeValuePair> mySupport;
    for (int j = 0; j < myI.nNodes; ++j) {
        if (y_start[j] > epsOpt) {
            nodeValuePair nVP;
            nVP.id = j;
            nVP.value = y_start[j];
            mySupport.push_back(nVP);
        }
    }
    vector<nodeValuePair> helper = mySupport;
    int obj = 0;

    
    vector<vector<nodeValuePair> > y_Supporter;
    y_Supporter.push_back(mySupport);

    int counter = 1;
    cout << "Error " << (myI.endP - myI.startP) << endl;

    vector<int> objectives;
    objectives.push_back(getMinMaxDistance(mySupport));

    while(counter < (myI.endP - myI.startP)) {
        //auto iter = helper.begin();
        int maxObj = INT_MAX;
        std::vector<nodeValuePair>::iterator iterDelete;
        
        for(int i = 0; i < helper.size(); ++i) {
            
            vector<nodeValuePair> deleter;
            copy(helper.begin(), helper.end(), back_inserter(deleter));
            std::vector<nodeValuePair>::iterator iter = (deleter.begin() + i);
            deleter.erase(iter);

            nodeValuePair nVP = *iter;
            //cout << nVP.id << endl;
            int distHelper = getMinMaxDistance(deleter);
            if (maxObj > distHelper) {
                maxObj = distHelper;
                std::vector<nodeValuePair>::iterator iterHelp = (helper.begin() + i);
                iterDelete = iterHelp;
            }
            //iter++;
            
        }
        objectives.push_back(maxObj);
        cout << "Obj: " << maxObj << endl;
        
        helper.erase(iterDelete);
        y_Supporter.push_back(helper);
        counter++;
    }
    
    IloNumVarArray vars(env);
    IloNumArray vals(env);
    int k = 0;
    for(int h = (myI.endP - myI.startP -1); h >= 0; --h) {
        vector<int> y_sol(myI.nNodes, 0);
        for (auto i:y_Supporter[h]) {
            y_sol[i.id] = 1;
        }
        for(int j = 0; j < myI.nNodes; ++j) {
            vars.add(y[k][j]);
            vals.add(y_sol[j]);
        }
        vars.add(z[k]);
        vals.add(objectives[h]);
        k++;
    }

    for (int i = 0; i < vars.getSize(); ++i) {
        if (vals[i] > epsOpt) {
            cout << vars[i] << ": " << vals[i] << endl;
        }
    }
    IloCplex::MIPStartEffort effort = IloCplex::MIPStartSolveMIP;
    cplex.addMIPStart(vars, vals, effort);
    
}

void MIPStart2(IloEnv env, IloIntVarArray2 y, IloIntVarArray z, IloCplex cplex) {

    cout << "Error" << endl; 
    set<int> helperSet;
    vector<nodeValuePair> openFacilities;
    for (int h = 0; h < myI.endP - myI.startP; ++h) {
        for (int j = 0; j < myI.nNodes; ++j) {
            if (y_solutions[h][j] > 0.9) {
                //cout << j << endl;
                if (helperSet.find(j) == helperSet.end()) {
                    helperSet.insert(j);
                    nodeValuePair nVP;
                    nVP.id = j;
                    nVP.distance = 1;
                    openFacilities.push_back(nVP);
                }
                else {
                    for (int k = 0; k < openFacilities.size(); ++k) {
                        //cout << openFacilities[k].id << "; " << j << endl;
                        if (openFacilities[k].id == j) {
                            openFacilities[k].distance ++;
                            //cout << openFacilities[k].distance << endl;
                            break;
                        }
                    }
                }
            }
        }
    }
    sort(openFacilities.begin(), openFacilities.end(), std::greater<nodeValuePair>());
    /*for (auto k:openFacilities) {
        cout << k.id << ": " << k.distance << endl;
    }*/

    // Pick the first myI.startP facilities to construct the smallest solution
    vector<nodeValuePair> firstSol(openFacilities.begin(), openFacilities.begin() + myI.startP);
    openFacilities.erase(openFacilities.begin(), openFacilities.begin() + myI.startP);
    int firstObj = getMinMaxDistance(firstSol);

    vector<vector<nodeValuePair> > solutions;
    vector<int> objectives;
    solutions.push_back(firstSol);
    objectives.push_back(firstObj);
    // Now add from the remaining facilities the best facility (smallest objective)
    vector<nodeValuePair> nextSol(firstSol);
    int nextObj = firstObj;
    while (solutions.size() < (myI.endP-myI.startP)) {
        vector<nodeValuePair> minSol = nextSol;
        int minObj = nextObj;
        for (auto k:openFacilities) {
            vector<nodeValuePair> helper(nextSol);
            helper.push_back(k);
            int helpObj = getMinMaxDistance(helper);
            if (helpObj < minObj) {
                minObj = helpObj;
                minSol = helper;
            }
        }
        nextSol = minSol;
        nextObj = minObj;
        solutions.push_back(nextSol);
        objectives.push_back(nextObj);
    }

    // Now add it as MIP Start solution
    vector<vector<int> > y_values;
    for (int h = 0; h < (myI.endP-myI.startP); ++h) {
        vector<int> helper(myI.nNodes, 0);
        for (auto k:solutions[h]) {
            helper[k.id] = 1;
        }
        y_values.push_back(helper);
    }

    IloNumVarArray vars(env);
    IloNumArray vals(env);
    for (int h = 0; h < (myI.endP-myI.startP); ++h) {
        for (int j = 0; j < myI.nNodes; ++j) {
            vars.add(y[h][j]);
            vals.add(y_values[h][j]);
        }
        vars.add(z[h]);
        vals.add(objectives[h]);
    }

    IloCplex::MIPStartEffort effort = IloCplex::MIPStartSolveMIP;
    cplex.addMIPStart(vars, vals, effort);

    for (int h = 0; h < myI.endP - myI.startP; ++h) {
        /*for (auto k:solutions[h]) {
            cout << k.id << "." << h << endl;
        }*/
        cout << objectives[h] << endl;
    }
}

void MIPStart3(IloEnv env, IloIntVarArray2 y, IloIntVarArray z, IloCplex cplex, vector<IloNumArray> y_solutions) {
    vector<nodeValuePair> mySupport;
    int horizon = y_solutions.size();
    for (int j = 0; j < myI.nNodes; ++j) {
        if (y_solutions[horizon-1][j] > epsOpt) {
            nodeValuePair nVP;
            nVP.id = j;
            nVP.value = y_start[j];
            mySupport.push_back(nVP);
        }
    }
    vector<nodeValuePair> helper = mySupport;
    //cout << mySupport.size() << endl;

    vector<vector<nodeValuePair> > y_Supporter;
    y_Supporter.push_back(mySupport);

    int counter = 1;

    vector<int> objectives;
    int firstObj = getMinMaxDistance(mySupport);
    objectives.push_back(firstObj);

    int maxObj = firstObj;
    int noImprove = 0;
    while(counter < horizon-1) {
        //cout << "Error" << endl;
        
        for (int j = 0; j < myI.nNodes; ++j) {
            //cout << noImprove << endl;
            if (noImprove < myI.nNodes) {
                vector<nodeValuePair> inserter;
                copy(helper.begin(), helper.end(), back_inserter(inserter));
                nodeValuePair nVP;
                nVP.id = j;
                nVP.value = 1;
                inserter.push_back(nVP);

                //cout << inserter.size() << endl;

                int distHelper = getMinMaxDistance(inserter);
                //cout << distHelper << endl;
                if (distHelper + epsOpt < maxObj) {
                        maxObj = distHelper;
                        y_Supporter.push_back(inserter);
                        objectives.push_back(distHelper);
                        helper = inserter;
                        counter++;
                        noImprove = 0;
                }       
                else {
                    noImprove++;
                }
            }
            else {
                vector<nodeValuePair> inserter;
                copy(helper.begin(), helper.end(), back_inserter(inserter));
                nodeValuePair nVP;
                nVP.id = j;
                nVP.value = 1;
                inserter.push_back(nVP);

                int distHelper = getMinMaxDistance(inserter);
                if (distHelper == maxObj) {
                    maxObj = distHelper;
                    y_Supporter.push_back(inserter);
                    objectives.push_back(distHelper);
                    helper = inserter;
                    counter++;
                    noImprove = 0;
                }       
            }
        }      
    }

    IloNumVarArray vars(env);
    IloNumArray vals(env);
    int k = 0;

    for (int h = 0; h < horizon; ++h) {
        vector<int> y_sol(myI.nNodes, 0);
        for (auto i:y_Supporter[h]) {
            y_sol[i.id] = 1;
        }
        for (int j = 0; j < myI.nNodes; ++j) {
            vars.add(y[h][j]);
            vals.add(y_sol[j]);
        }
        vars.add(z[k]);
        vals.add(objectives[h]);
    }

    for (int i = 0; i < vars.getSize(); ++i) {
        if (vals[i] > epsOpt) {
            cout << vars[i] << ": " << vals[i] << endl;
        }
    }

    IloCplex::MIPStartEffort effort = IloCplex::MIPStartSolveMIP;
    cplex.addMIPStart(vars, vals, effort);

}

vector<nodeValuePair2> seperationPC(IloNumArray y_lb, int LB, set<int> bigI) {
    //timer.resume();
    int nNodes = myI.nNodes;
    vector<nodeValuePair> openNodes;
    
    for (int j = 0; j < nNodes; ++j) {
        if (y_lb[j] > 0) {
            nodeValuePair nVP;
            nVP.id = j;
            nVP.value = y_lb[j];
            openNodes.push_back(nVP);
        }
    }

    vector<nodeValuePair2> violatedNodes;
    for (auto cust:bigI) {
        vector<nodeValuePair> violatedFacilities;
        for(auto j:openNodes) {
            nodeValuePair nVP2;
            nVP2.id = j.id;
            nVP2.value = j.value;
            nVP2.distance = max(getDistance(cust, j.id), LB);
            /*if (getDistance(cust,j.id) > LB) {
                nVP2.distance = getDistance(cust, j.id);
            }
            else {
                nVP2.distance = LB;
            }*/
            violatedFacilities.push_back(nVP2);
        }
        sort(violatedFacilities.begin(), violatedFacilities.end(), std::less<nodeValuePair>());


        double valueSum = 0;
        int critLocation = -1;
        int critDistance = -1;

        for (auto i:violatedFacilities) {
            //cout << i.distance << endl;
            //valueSum += i.value;
            if (valueSum + i.value < 1) {
                valueSum += i.value;
            } else {
                critDistance = i.distance;
                critLocation = i.id;
                break;
            }
            //cout << endl;
        }
        
        double violation = critDistance;

        for (auto j:violatedFacilities) {
            int distance = getDistance(cust, j.id);
            double coefficient = critDistance - max(LB, (int) getDistance(cust, j.id));
            if (j.value == 1 && getDistance(cust, j.id) < critDistance) {
                violation -= coefficient;
            }
        }

        if (violation > 0 + epsOpt) {
            nodeValuePair2 nVP2;
            nVP2.cust = cust;
            nVP2.fac = critLocation;
            nVP2.violation = violation;
            nVP2.criticalDistance = critDistance; // Distance saves now the location index
            violatedNodes.push_back(nVP2);
        }
    }

    sort(violatedNodes.begin(), violatedNodes.end(), std::greater<nodeValuePair2>());

    //timer.stop();
    sepCounter++;

    return violatedNodes;

}

cutViolation generateCut(nodeValuePair2 violatedNode, IloIntVarArray y, IloNumArray y_lb, IloEnv myEnv, int LB) {
    //cout << "Cut generation called" << endl;
    IloExpr myCut(myEnv);
    cutViolation cV;
    double violation = violatedNode.criticalDistance;
    myCut += violatedNode.criticalDistance;

    //vector<int> sol = {2563,3398,3452};
    //vector<int> y_sol(myI.nNodes);
    //for (auto j:sol) {
    //    y_sol[j] = 1;
    //}
    double RHS2 = violatedNode.criticalDistance;

    for (int j = 0; j < myI.nNodes; ++j) {
        int distance = getDistance(violatedNode.cust, j);
        if (distance < violatedNode.criticalDistance) {
            double coefficient = violatedNode.criticalDistance - max(LB, distance);
            violation -= coefficient*y_lb[j];
            myCut -= coefficient*y[j];
            //RHS2 -= coefficient*y_sol[j];
        }
    }
    
    //cout << "RHS2: " << RHS2 << endl;

    cV.myCut = myCut;
    cV.violation = violation;
    cV.RHS2 = RHS2;

    /*if (cV.RHS2 == 6377) {
        cout << "RHS2  == 6377" << endl;
    }*/

    //cout << "Cut: " << cV.violation << "; " << cV.RHS2 << endl;
    
    return cV;
}

vector<nodeValuePair2> seperationNpC(IloNumArray2 y_lb, vector<int> LB, int h, set<int> bigI) {
    int nNodes = myI.nNodes;
    vector<nodeValuePair> openNodes;

    for (int j = 0; j < nNodes; ++j) {
        if (y_lb[h][j] > 0) {
            nodeValuePair nVP;
            nVP.id = j;
            nVP.value = y_lb[h][j];
            nVP.horizon = h;
            openNodes.push_back(nVP);
        }
    }

    vector<nodeValuePair2> violatedNodes;
    for (auto cust:bigI) {
        vector<nodeValuePair> violatedFacilities;
        for(auto j:openNodes) {
            nodeValuePair nVP2;
            nVP2.id = j.id;
            nVP2.value = j.value;
            nVP2.horizon = h;
            if (getDistance(cust,j.id) > LB[h]) {
                nVP2.distance = getDistance(cust, j.id);
            }
            else {
                nVP2.distance = LB[h];
            }
            violatedFacilities.push_back(nVP2);
        }
        sort(violatedFacilities.begin(), violatedFacilities.end(), std::less<nodeValuePair>());

        // Now calculate the violation for every customer

        int iter = 0;
        double valueSum = violatedFacilities[iter].value;

        while (valueSum < 1) {
            iter++;
            valueSum += violatedFacilities[iter].value;
        }

        int critLocation = violatedFacilities[iter].id;
        int critDistance = violatedFacilities[iter].distance;
        double violation = critDistance;

        for (auto j:violatedFacilities) {
            double coefficient = critDistance - max((int) LB[h], (int) getDistance(cust, j.id));
            if (j.value == 1 && getDistance(cust, j.id) < critDistance) {
                violation -= coefficient;
            }
        }

        if (violation > 0) {
            nodeValuePair2 nVP2;
            nVP2.cust = cust;
            nVP2.fac = critLocation;
            nVP2.violation = violation;
            nVP2.horizon = h;
            nVP2.criticalDistance = critDistance; // Distance saves now the location index
            violatedNodes.push_back(nVP2);
        }
    }
    sort(violatedNodes.begin(), violatedNodes.end(), std::greater<nodeValuePair2>());

    return violatedNodes;
}

cutViolation generateNCut(nodeValuePair2 violatedNode, IloIntVarArray2 y, IloNumArray2 y_lb, IloEnv myEnv, vector<int> LB, int h) {
    IloExpr myCut(myEnv);
    cutViolation cV;
    double violation = violatedNode.criticalDistance;
    myCut += violatedNode.criticalDistance;

    for (int j = 0; j < myI.nNodes; ++j) {
        int distance = getDistance(violatedNode.cust, j);
        if (distance < violatedNode.criticalDistance) {
            double coefficient = violatedNode.criticalDistance - max(LB[h], distance);
            violation -= coefficient*y_lb[h][j];
            myCut -= coefficient*y[h][j];
        }
    }

    cV.myCut = myCut;
    cV.violation = violation;
    
    return cV;
}

void initSetfC(int p) {
    
    for (int cust = 0; cust < myI.nNodes; cust++) {
        bigI.insert(cust);
    }
    set<int> inHat = bigI;
    iHat.clear();

    int initCust = rand()%myI.nNodes;
    iHat.insert(initCust);
    inHat.erase(initCust);

    while(iHat.size() <= p) {
        int maxID = -1;
        int maxDist = -1;
        for (auto i:inHat) {
            int minDist = INT_MAX;
            for (auto j:iHat) {
                int distHelper = getDistance(i,j);
                if (distHelper < minDist) {
                    minDist = distHelper;
                }
            }
            if (minDist > maxDist) {
                maxDist = minDist;
                maxID = i;
            }
        }
        if (maxID != -1) {
            iHat.insert(maxID);
            inHat.erase(maxID);
        }
    }

    initLB = INT_MAX;
    for (auto i:iHat) {
        for (auto j:bigI) {
            if (i == j) {
                continue;
            }
            int distHelper = getDistance(i, j);
            if (distHelper < initLB) {
                initLB = distHelper;
            }
        }
    }
    
    /*
    for (int i = 0; i < myI.nNodes; ++i) {
        inonHat.insert(i);
    }

    int initCust = rand()%myI.nNodes;
    inonHat.erase(initCust);
    iHat.insert(initCust);

    while (iHat.size() <= myI.startP) {
        vector<nodeValuePair> customerDistance;
        int maxDist = 0;
        int maxCust = -1;
        for (int i:inonHat) {
            int minDist = INT_MAX;
            for (int j:iHat) {
                int distHelp = getDistance(i,j);
                if (distHelp < minDist) {
                    minDist = distHelp;
                }
            }
            if (minDist > maxDist) {
                maxDist = minDist;
                maxCust = i;
            }
        }
        inonHat.erase(maxCust);
        iHat.insert(maxCust);
        //cout << iHat.size() << endl;
    }

    int minDist = INT_MAX;
    for (auto i:iHat) {
        for (int j = 0; j < myI.nNodes; ++j) {
            if (i == j) {
                continue;
            }
            int distHelper = getDistance(i,j);
            if (distHelper < minDist) {
                minDist = distHelper;
            }
        }
    }
    //initLB = minDist;

    iHatLazy = iHat;
    inonHatLazy = inonHat;
    */
    //set_difference(bigI.begin(), bigI.end(), iHat.begin(), iHat.end(), std::inserter(inonHat2, inonHat2.begin()));
}

void setCPLEXParameters(IloCplex myCplex)
{

	myCplex.setParam(IloCplex::Param::MIP::Tolerances::AbsMIPGap,0);
	//myCplex.setParam(IloCplex::Param::WorkMem, params.memlimit);
	//myCplex.setParam(IloCplex::Param::MIP::Limits::TreeMemory, params.memlimit);
	myCplex.setParam(IloCplex::Param::MIP::Strategy::File, 3);
	myCplex.setParam(IloCplex::Param::ClockType, 1);

	myCplex.setParam(IloCplex::Param::Threads,params.threads);
	myCplex.setParam(IloCplex::Param::TimeLimit,params.timelimit);
	myCplex.setParam(IloCplex::Param::MIP::Tolerances::MIPGap,0.0);

	myCplex.setParam(IloCplex::Param::MIP::Cuts::Cliques, params.cplexcuts-1);
	myCplex.setParam(IloCplex::Param::MIP::Cuts::FlowCovers, params.cplexcuts-1);
	myCplex.setParam(IloCplex::Param::MIP::Cuts::Gomory, params.cplexcuts-1);
	myCplex.setParam(IloCplex::Param::MIP::Cuts::ZeroHalfCut, params.cplexcuts-1);
	myCplex.setParam(IloCplex::Param::MIP::Cuts::LiftProj, params.cplexcuts-1);
	myCplex.setParam(IloCplex::Param::MIP::Cuts::GUBCovers, params.cplexcuts-1);
	myCplex.setParam(IloCplex::Param::MIP::Cuts::Covers, params.cplexcuts-1);
	myCplex.setParam(IloCplex::Param::MIP::Cuts::Disjunctive , params.cplexcuts-1);
	myCplex.setParam(IloCplex::Param::MIP::Cuts::Implied , params.cplexcuts-1);
	myCplex.setParam(IloCplex::Param::MIP::Cuts::MCFCut , params.cplexcuts-1);
	myCplex.setParam(IloCplex::Param::MIP::Cuts::MIRCut , params.cplexcuts-1);
	myCplex.setParam(IloCplex::Param::MIP::Cuts::PathCut, params.cplexcuts-1);

	//myCplex.setParam(IloCplex::Param::RootAlgorithm, 4);
}

int getMinMaxDistance(vector<nodeValuePair> helper) {
    int maxDistance = 0;
    for (int i = 0; i < myI.nNodes; ++i) {
        int minDistance = INT_MAX;
        for (auto j:helper) {
            int distanceHelper = getDistance(i, j.id);
            if (distanceHelper < minDistance) {
                minDistance = distanceHelper;
            }
        }
        if (minDistance > maxDistance) {
            maxDistance = minDistance;
        }
    }
    return maxDistance;
}

void printSTATs(IloEnv masterEnv, Instance myI, vector<int> P) {
    
    //std::string filename = "test";
    std::string filename(boost::filesystem::path(params.file).stem().string());
    std::string model = "n_ell";
    
    //boost::timer::cpu_times times = timer.elapsed();

    sec seconds = boost::chrono::nanoseconds(timer.elapsed().wall);


    int start;
    int end;

    if (params.instanceformat == 1) {
        start = myI.startP;
        end = myI.startP + params.endP;
    }
    if (params.instanceformat == 2) {
        start = params.startP;
        end = params.endP;
    }
    if (objvalue == bestLB) {
        optimality = 1;
    }

    masterEnv.out() << "STAH, fullname, filename, instanceformat, nNodes, start, end, model, cputime, roottime, optimality, objvalue, rootbound, rootUB, bestUB, bestLB, timelimit" << endl;
    masterEnv.out() << "STAT,"
                    <<params.file<<","
                    <<filename<<","
                    <<params.instanceformat<<","
                    <<myI.nNodes<<","
                    <<start<<","
                    <<end<<","
                    <<model<<","
                    <<seconds.count()<<","
                    <<roottime<<","
                    <<optimality<<","
                    <<objvalue<<","
                    <<rootbound<<","
                    <<rootUB<<","
                    <<bestUB<<","
                    <<bestLB<<","
                    <<params.timelimit<<endl;
}

/*bool checkHSol() {

}*/