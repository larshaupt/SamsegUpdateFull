/**
 * @file  QdecGlmDesign.cpp
 * @brief Contains the data and functions associated with a GLM design run
 *
 * Contains the data and functions associated with the selected GLM input
 * (design) parameters: selected discrete and continuous factors, measure,
 * hemi, smoothness level and design name.  Functions exist to create derived
 * data: .fsgd file, contrast vectors in .mat file format, and the 'y' input
 * data file (concatenated subjects file).
 */
/*
 * Original Author: Nick Schmansky
 * CVS Revision Info:
 *    $Author: nicks $
 *    $Date: 2008/01/17 02:40:16 $
 *    $Revision: 1.10 $
 *
 * Copyright (C) 2007-2008,
 * The General Hospital Corporation (Boston, MA).
 * All rights reserved.
 *
 * Distribution, usage and copying of this software is covered under the
 * terms found in the License Agreement file named 'COPYING' found in the
 * FreeSurfer source code root directory, and duplicated here:
 * https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferOpenSourceLicense
 *
 * General inquiries: freesurfer@nmr.mgh.harvard.edu
 * Bug reports: analysis-bugs@nmr.mgh.harvard.edu
 *
 */

#include <errno.h>
#include <sys/stat.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "QdecGlmDesign.h"


// Constructors/Destructors
//

QdecGlmDesign::QdecGlmDesign ( QdecDataTable* iDataTable )
{
  assert( iDataTable );
  this->mDataTable = iDataTable;
  this->mbValid = false;
  this->msName = "Untitled";
  this->msMeasure = "thickness";
  this->msHemi = "lh";
  this->mSmoothness = 10;
  if( NULL == getenv("SUBJECTS_DIR") )
  {
    fprintf( stderr, "SUBJECTS_DIR not defined!\n" );
    exit(1);
  }
  this->mfnSubjectsDir = getenv("SUBJECTS_DIR");
  this->msAverageSubject = "fsaverage";
  this->mfnFsgdfFile = "qdec.fsgd";
  this->mfnYdataFile = "y.mgh";
  this->mfnDefaultWorkingDir = "";
  if( NULL != getenv("QDEC_WORKING_DIR") )
  {
    this->mfnDefaultWorkingDir = getenv("QDEC_WORKING_DIR");
  }
  if( "" == this->mfnDefaultWorkingDir )
  {
    this->mfnDefaultWorkingDir = this->mfnSubjectsDir;
    this->mfnDefaultWorkingDir += "/qdec";
  }
  this->mfnWorkingDir = this->mfnDefaultWorkingDir;
}

QdecGlmDesign::~QdecGlmDesign ( )
{
  while (mContrasts.size() != 0)
  {
    delete mContrasts.back();
    mContrasts.pop_back();
  }
}

//
// Methods
//


/**
 * Returns true if this design is valid (input parameters have been set and
 * mri_glmfit input data created, ie. Create() has been called successfully)
 * @return bool
 */
bool QdecGlmDesign::IsValid ( )
{
  return this->mbValid;
}


/**
 * Initializes the design with the given design parameters.
 * @return int
 * @param  iDataTable
 * @param  isName
 * @param  isFirstDiscreteFactor
 * @param  isSecondDiscreteFactor
 * @param  isFirstContinuousFactor
 * @param  isSecondContinuousFactor
 * @param  isMeasure
 * @param  isHemi
 * @param  iSmoothnessLevel
 * @param  iProgressUpdateGUI
 */
int QdecGlmDesign::Create ( QdecDataTable* iDataTable,
                            const char* isName,
                            const char* isFirstDiscreteFactor,
                            const char* isSecondDiscreteFactor,
                            const char* isFirstContinuousFactor,
                            const char* isSecondContinuousFactor,
                            const char* isMeasure,
                            const char* isHemi,
                            int iSmoothnessLevel,
                            ProgressUpdateGUI* iProgressUpdateGUI )
{

  this->mbValid = false; // set to true if successful

  // this is the callers GUI app, but it can be null
  this->mProgressUpdateGUI = iProgressUpdateGUI;
  if( this->mProgressUpdateGUI )
  {
    this->mProgressUpdateGUI->BeginActionWithProgress("Create GLM design..." );
  }

  // delete any prior runs
  mDiscreteFactors.clear();    // We don't own the data in these
  mContinuousFactors.clear();  // containers; QdecDataTable does

  while (mContrasts.size() != 0)
  {
    delete mContrasts.back();
    mContrasts.pop_back();
  }

  // begin absorbing input parameters

  assert( iDataTable );
  this->mDataTable = iDataTable;

  //
  // Set up the params for the processing. Check the input to make
  // sure everything we expect is present.
  //

  // Copy the input parameters into our design storage
  if( this->mProgressUpdateGUI )
  {
    this->mProgressUpdateGUI->UpdateProgressMessage( "Extracting design..." );
    this->mProgressUpdateGUI->UpdateProgressPercent( 10 );
  }

  this->msName = isName;
  this->msMeasure = isMeasure;
  this->msHemi = isHemi;
  this->mSmoothness = iSmoothnessLevel;
  QdecFactor* qf;
  if ( (NULL != isFirstDiscreteFactor) &&
       (strcmp(isFirstDiscreteFactor,"none")) )
  {
    qf = this->mDataTable->GetFactor( isFirstDiscreteFactor );
    if( NULL == qf )
    {
      fprintf( stderr,"ERROR: QdecGlmDesign::Create: bad factor!\n" );
      mDataTable->Dump( stderr );
      return -1;
    }
    assert( qf->IsDiscrete() );
    this->mDiscreteFactors.push_back( qf );
  }
  if ( (NULL != isSecondDiscreteFactor) &&
       (strcmp(isSecondDiscreteFactor,"none")) )
  {
    qf = this->mDataTable->GetFactor( isSecondDiscreteFactor );
    if( NULL == qf )
    {
      fprintf( stderr,"ERROR: QdecGlmDesign::Create: bad factor!\n" );
      return -1;
    }
    assert( qf->IsDiscrete() );
    this->mDiscreteFactors.push_back( qf );
  }
  if ( (NULL != isFirstContinuousFactor) &&
       (strcmp(isFirstContinuousFactor,"none")) )
  {
    qf = this->mDataTable->GetFactor( isFirstContinuousFactor );
    if( NULL == qf )
    {
      fprintf( stderr,"ERROR: QdecGlmDesign::Create: bad factor!\n" );
      return -1;
    }
    assert( qf->IsContinuous() );
    this->mContinuousFactors.push_back( qf );
  }
  if ( (NULL != isSecondContinuousFactor) &&
       (strcmp(isSecondContinuousFactor,"none")) )
  {
    qf = this->mDataTable->GetFactor( isSecondContinuousFactor );
    if( NULL == qf )
    {
      fprintf( stderr,"ERROR: QdecGlmDesign::Create: bad factor!\n" );
      return -1;
    }
    assert( qf->IsContinuous() );
    this->mContinuousFactors.push_back( qf );
  }
  if ( 0 == (this->mDiscreteFactors.size() + this->mContinuousFactors.size()) )
  {
    fprintf( stderr,"ERROR: QdecGlmDesign::Create: zero factors!\n" );
    return -1;
  }

  // Create the fsgd and contrast files, writing these to the working dir
  if( this->mProgressUpdateGUI )
  {
    this->mProgressUpdateGUI->UpdateProgressMessage
      ( "Saving configuration design..." );
    this->mProgressUpdateGUI->UpdateProgressPercent( 20 );
  }

  int err = mkdir( this->mfnWorkingDir.c_str(), 0777);
  if( err != 0 && errno != EEXIST )
  {
    fprintf( stderr,
             "ERROR: QdecGlmDesign::Create: could not create directory %s\n",
             this->mfnWorkingDir.c_str());
    return(-2);
  }

  if( this->mProgressUpdateGUI )
  {
    this->mProgressUpdateGUI->EndActionWithProgress();
  }

  // Make all our contrasts.
  this->GenerateContrasts();

  this->mbValid = true; // success
  return 0;
}


/**
 * @return string
 */
string QdecGlmDesign::GetName ( )
{
  return this->msName;
}


/**
 * @return string
 */
string QdecGlmDesign::GetHemi ( )
{
  return this->msHemi;
}


/**
 * @return string
 */
string QdecGlmDesign::GetMeasure ( )
{
  return this->msMeasure;
}


/**
 * @return int
 */
int QdecGlmDesign::GetSmoothness ( )
{
  return this->mSmoothness;
}


/**
 * @return string
 */
string QdecGlmDesign::GetSubjectsDir ( )
{
  return this->mfnSubjectsDir;
}


/**
 * @param const char*
 */
int QdecGlmDesign::SetSubjectsDir ( const char* ifnSubjectsDir )
{
  this->mfnSubjectsDir = ifnSubjectsDir; 
  if ( setenv( "SUBJECTS_DIR",  ifnSubjectsDir, 1) ) 
  {
    printf( "ERROR: failure setting SUBJECTS_DIR to '%s'\n", ifnSubjectsDir );
    return -1;
  }
  printf( "SUBJECTS_DIR is '%s'\n", ifnSubjectsDir );
  return 0;
}


/**
 * @return string
 */
string QdecGlmDesign::GetAverageSubject ( )
{
  return this->msAverageSubject;
}


/**
 * @param const char*
 */
void QdecGlmDesign::SetAverageSubject ( const char* isAverageSubject )
{
  this->msAverageSubject = isAverageSubject;
}


/**
 * returns the pathname to the fsgd file required by mri_glmfit.
 * @return string
 */
string QdecGlmDesign::GetFsgdFileName ( )
{
  string tmp = this->mfnWorkingDir;
  tmp += "/";
  tmp += this->mfnFsgdfFile;
  return tmp;
}


/**
 * returns the pathname to the input data, 'y', required by mri_glmfit.
 * @return string
 */
string QdecGlmDesign::GetYdataFileName ( )
{
  string tmp = this->mfnWorkingDir;
  tmp += "/";
  tmp += this->mfnYdataFile;
  return tmp;
}


/**
 * @return vector< string >
 */
vector< string > QdecGlmDesign::GetContrastNames ( )
{
  vector<string> tmp;
  for( unsigned int i=0; i < this->mContrasts.size(); i++)
  {
    tmp.push_back( this->mContrasts[i]->GetName() );
  }
  return tmp;
}


/**
 * @return vector< string >
 */
vector< string > QdecGlmDesign::GetContrastQuestions ( )
{
  vector<string> tmp;
  for( unsigned int i=0; i < this->mContrasts.size(); i++)
  {
    tmp.push_back( this->mContrasts[i]->GetQuestion() );
  }
  return tmp;
}


/**
 * @return vector< string >
 */
vector< string > QdecGlmDesign::GetContrastFileNames ( )
{
  vector<string> tmp;
  for( unsigned int i=0; i < this->mContrasts.size(); i++)
  {
    tmp.push_back( this->mContrasts[i]->GetDotMatFileName() );
  }
  return tmp;
}


/**
 * @return string
 */
string QdecGlmDesign::GetDefaultWorkingDir ( )
{
  return this->mfnDefaultWorkingDir;
}


/**
 * @return string
 */
string QdecGlmDesign::GetWorkingDir ( )
{
  return this->mfnWorkingDir;
}


/**
 * @return int
 * @param  isPathName
 */
int QdecGlmDesign::SetWorkingDir (const char* isPathName )
{
  this->mfnWorkingDir = isPathName;
  return 0;
}


/**
 * @return ProgressUpdateGUI*
 */
ProgressUpdateGUI* QdecGlmDesign::GetProgressUpdateGUI ( )
{
  return this->mProgressUpdateGUI;
}

/**
 * SetExcludeSubjectID( const char* isSubjecID, bool ibExclude ) -
 * sets a subject ID's exclusion status. If excluded, it will not be
 * included when writing the fsgd and ydata files. This function modifies the
 * maExcludedSubjects set, adding the subject ID if ibExclude is true
 * and removing it if not.
 * param const char* isSubjectID
 * param bool ibExclude
 */
void QdecGlmDesign::SetExcludeSubjectID ( const char* isSubjectID, 
                                          bool ibExclude )
{
  assert( isSubjectID );
  
  if( ibExclude ) {
    maExcludedSubjects.insert( string(isSubjectID) );
  } else {
    if( maExcludedSubjects.find( string(isSubjectID) ) != 
        maExcludedSubjects.end() ) {
      maExcludedSubjects.erase( string(isSubjectID) );
    }
  }
}

/**
 * GetExcludeSubjectID ( const char* isSubjecID ) -
 * Returns a subject ID's exclusion status.
 * param const char* isSubjectID
 */
bool QdecGlmDesign::GetExcludeSubjectID ( const char* isSubjectID )
{
  assert( isSubjectID );
  
  if( maExcludedSubjects.find( string(isSubjectID) ) != 
      maExcludedSubjects.end() )
    return true;
  else
    return false;
}

/**
 * SetExcludeSubjectsFactorGT
 */
void  QdecGlmDesign::SetExcludeSubjectsFactorGT ( const char* isFactorName,
                                                  double inExcludeGT,
                                                  bool ibExclude )
{
  assert( isFactorName );

  vector< QdecSubject* > subjs = this->mDataTable->GetSubjects();
  unsigned int nInputs = subjs.size();
  for (unsigned int m=0; m < nInputs; m++)
  {
    if (subjs[m]->GetContinuousFactor( isFactorName ) > inExcludeGT)
    {
      this->SetExcludeSubjectID( subjs[m]->GetId().c_str(), ibExclude );
    }
  }
}

/**
 * SetExcludeSubjectsFactorLT
 */
void  QdecGlmDesign::SetExcludeSubjectsFactorLT ( const char* isFactorName,
                                                  double inExcludeLT,
                                                  bool ibExclude )
{
  assert( isFactorName );

  vector< QdecSubject* > subjs = this->mDataTable->GetSubjects();
  unsigned int nInputs = subjs.size();
  for (unsigned int m=0; m < nInputs; m++)
  {
    if (subjs[m]->GetContinuousFactor( isFactorName ) < inExcludeLT)
    {
      this->SetExcludeSubjectID( subjs[m]->GetId().c_str(), ibExclude );
    }
  }
}

/**
 * ClearAllExcludedSubjects
 */
void  QdecGlmDesign::ClearAllExcludedSubjects ( )
{
  vector< QdecSubject* > subjs = this->mDataTable->GetSubjects();
  unsigned int nInputs = subjs.size();
  for (unsigned int m=0; m < nInputs; m++)
  {
    this->SetExcludeSubjectID( subjs[m]->GetId().c_str(), 0 );
  }
}

/**
 * GetNumberOfExcludedSubjects
 */
int QdecGlmDesign::GetNumberOfExcludedSubjects ( )
{
  return this->maExcludedSubjects.size();
}


/**
 * GetNumberOfClasses( ) - returns the number of classes for the design.
 * The number of classes is just all the combinations of all
 * the levels for the discrete factors.
 * @return int
 */
int QdecGlmDesign::GetNumberOfClasses( )
{
  int nClasses = 1;
  for (unsigned int i=0; i < this->mDiscreteFactors.size(); i++)
  {
    nClasses *= this->mDiscreteFactors[i]->GetLevelNames().size();
  }
  return(nClasses);
}


/**
 * GetNumberOfRegressors() - returns the number of regressors for the
 * given design.
 */
int QdecGlmDesign::GetNumberOfRegressors( )
{
  int nReg = this->GetNumberOfClasses() * 
    ( this->GetNumberOfContinuousFactors() + 1 );
  return(nReg);
}


/**
 * GetLevels2ClassName() - returns the class name given that
 * the 1st factor is at nthlevels[0],
 * the 2nd factor is at nthlevels[1], ...
 * The class name is created by appending
 *   Factor1NameLevelName-Factor2NameLevelName...
 */
string QdecGlmDesign::GetLevels2ClassName ( unsigned int* nthlevels )
{
  string ClassName = "";
  unsigned int df = 0;
  unsigned int ndf = this->GetNumberOfDiscreteFactors();

  for ( unsigned int f=0; f < ndf; f++ )
  {
    QdecFactor* F = this->mDiscreteFactors[f];
    unsigned int nLevels = F->GetLevelNames().size();
    assert( nLevels );
    if ( nthlevels[df] >= nLevels )
    {
      fprintf
        (stderr,
         "ERROR: QdecGlmDesign::GetLevels2ClassName: "
         "factor %s, level %d >= nlevels = %d\n",
         F->GetFactorName().c_str(),
         nthlevels[df],
         nLevels);
      break;
    }
    vector< string > levelNames = F->GetLevelNames();
    ClassName += F->GetFactorName();
    ClassName += levelNames[nthlevels[df]];
    if (ndf > 1 && df < ndf-1) ClassName += "-";
    df++;
  }

  return( ClassName );
}



/**
 * Creates an FSGD File from the given design and writes it to the working dir.
 * @return int
 */
int QdecGlmDesign::WriteFsgdFile ( )
{
  if( !this->IsValid() ) {
    fprintf( stderr, "ERROR: QdecGlmDesign::WriteFsgdFile: "
             "Design parameters not valid.\n" );
    return(-1);
  }
  
  string fsgdFile = this->GetFsgdFileName();

  FILE *fp = fopen(fsgdFile.c_str(),"w");
  if (fp == NULL)
  {
    fprintf( stderr, "ERROR: QdecGlmDesign::WriteFsgdFile: "
             "could not open %s for writing\n",
             fsgdFile.c_str() );
    return(-1);
  }

  fprintf(fp,"GroupDescriptorFile 1\n");
  fprintf(fp,"Title %s\n",this->GetName().c_str());
  fprintf(fp,"MeasurementName %s\n",this->GetMeasure().c_str());

  unsigned int nDiscreteFactors = this->GetNumberOfDiscreteFactors();
  unsigned int nClasses = this->GetNumberOfClasses();
  if ( nDiscreteFactors > 0)
  {
    unsigned int* levels = 
      (unsigned int *) calloc( nDiscreteFactors, sizeof(unsigned int) );
    for (unsigned int nthclass = 0; nthclass < nClasses; nthclass++)
    {
      string ClassName = this->GetLevels2ClassName( levels );
      fprintf(fp,"Class %s\n",ClassName.c_str());
      unsigned int f = 0;
      levels[f]++;
      while (1)
      {
        if (levels[f] == this->mDiscreteFactors[f]->GetLevelNames().size())
        {
          levels[f] = 0;
          f++;
          if ( f == nDiscreteFactors ) break;
          levels[f]++;
        }
        else break;
      }
    }
  }
  else fprintf(fp,"Class Main\n");

  unsigned int nContinuousFactors = this->GetNumberOfContinuousFactors();
  if ( nContinuousFactors > 0 )
  {
    fprintf(fp,"Variables ");
    for (unsigned int f=0; f < nContinuousFactors; f++)
    {
      fprintf( fp, "%s ",
               this->mContinuousFactors[f]->GetFactorName().c_str() );
    }
    fprintf( fp, "\n" );
  }

  vector< QdecSubject* > subjs = this->mDataTable->GetSubjects();
  unsigned int nInputs = subjs.size();
  for (unsigned int m=0; m < nInputs; m++)
  {
    // If this name is in our list of subject exclusions, skip it.
    if( maExcludedSubjects.find( subjs[m]->GetId() ) !=
        maExcludedSubjects.end() ) {
      continue;
    }

    fprintf( fp, "Input %s ", subjs[m]->GetId().c_str() );
    if ( nDiscreteFactors > 0 )
    {
      string ClassName = "";
      unsigned int df = 0;
      unsigned int ndf = this->GetNumberOfDiscreteFactors();
      for (unsigned int f=0; f < ndf; f++)
      {
        string factorName = mDiscreteFactors[f]->GetFactorName();
        ClassName += factorName;
        ClassName += subjs[m]->GetDiscreteFactor( factorName.c_str() );
        if (ndf > 1 && df < ndf-1) ClassName += "-";
        df++;
      }
      fprintf(fp,"%s ",ClassName.c_str());
    }
    else fprintf( fp,"Main " );
    if ( nContinuousFactors > 0 )
    {
      for (unsigned f=0; f < nContinuousFactors; f++)
      {
        const char* factorName = 
          this->mContinuousFactors[f]->GetFactorName().c_str();
        fprintf( fp, "%lf ", subjs[m]->GetContinuousFactor( factorName ) );
      }
    }
    fprintf( fp, "\n" );
  }

  fclose(fp);

  return(0);
}



/**
 * Creates Contrast objects based on the selected factors.
 * Stores them in our 'contrast' object container.
 * @return int
 */
int QdecGlmDesign::GenerateContrasts ( )
{
  char tmpstr[2000];

  while (mContrasts.size() != 0)
  {
    delete mContrasts.back();
    mContrasts.pop_back();
  }

  /*----------------------------------------------------------*/
  unsigned int ndf = this->GetNumberOfDiscreteFactors();
  if (ndf > 2)
  {
    fprintf(stderr,
            "ERROR: QdecGlmDesign::GenerateContrasts: "
            "number of discrete factors = %d > 2\n",
            ndf);
    return -1;
  }
  unsigned int ncf = this->GetNumberOfContinuousFactors();
  unsigned int nreg = this->GetNumberOfRegressors();

  /*----------------------------------------------------------*/
  if (ndf == 0)
  {
    int nc = ncf+1;
    for (int nthcf = 0; nthcf < nc; nthcf++)
    {
      vector< double > contrast;
      for (unsigned int i=0; i < nreg; i++) contrast.push_back( 0.0 );
      assert( contrast.size() == nreg );
      contrast[nthcf] = 1;
      string name = "";
      string question = "";
      if (nthcf == 0)
      {
        sprintf(tmpstr,
                "Avg-Intercept-%s",
                this->msMeasure.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,
                "Does the average %s differ from zero?",
                this->msMeasure.c_str());
        question = strdup(tmpstr);
      }
      else if (ncf == 1)
      {
        QdecFactor* contFactor = this->mContinuousFactors[nthcf-1];
        string contFactorName = contFactor->GetFactorName();
        sprintf(tmpstr,
                "Avg-%s-%s-Cor",
                this->msMeasure.c_str(),
                contFactorName.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,
                "Does the correlation between %s and %s differ from zero?",
                this->msMeasure.c_str(),
                contFactorName.c_str());
        question = strdup(tmpstr);
      }
      else if (ncf == 2)
      {
        QdecFactor* contFactor = this->mContinuousFactors[nthcf-1];
        string contFactorName = contFactor->GetFactorName();
        QdecFactor* otherFactor;
        if (nthcf == 1) otherFactor = this->mContinuousFactors[nthcf];
        else otherFactor = this->mContinuousFactors[nthcf-2];
        string otherFactorName = otherFactor->GetFactorName();
        sprintf(tmpstr,
                "Avg-%s-%s-Cor",
                this->msMeasure.c_str(),
                contFactorName.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,
                "Does the correlation between %s and %s, accounting for %s, "
                "differ from zero?",
                this->msMeasure.c_str(),
                contFactorName.c_str(),
                otherFactorName.c_str());
        question = strdup(tmpstr);
      }
      QdecContrast* newContrast = new QdecContrast ( contrast,
                                                     name,
                                                     question );
      this->mContrasts.push_back( newContrast );
    }
  }

  /*------------------------------------------------------------*/
  if (ndf == 1)
  {
    // Two types of tests/contrasts: MainEffect and SimpleMainEffect
    QdecFactor* df1 = this->mDiscreteFactors[0];
    string df1Name = df1->GetFactorName();
    int nl1 = df1->GetLevelNames().size();
    if (nl1 != 2)
    {
      fprintf(stderr, "ERROR: QdecGlmDesign::GenerateContrasts: "
              "factor 1 must have 2 levels\n");
      return -1;
    }
    vector<string> levelNames = df1->GetLevelNames();
    const char* df1l1name = levelNames[0].c_str();
    const char* df1l2name = levelNames[1].c_str();

    // Do the Main Effects first
    for (unsigned int nthcf = 0; nthcf < (ncf+1); nthcf++)
    {
      vector< double > contrast;
      for (unsigned int i=0; i < nreg; i++) contrast.push_back( 0.0 );
      assert( contrast.size() == nreg );
      int a = 2*nthcf;
      contrast[a] = 1;
      contrast[a+1] = 1;
      string name = "";
      string question = "";
      if (nthcf == 0)
      {
        sprintf(tmpstr,
                "Avg-Intercept-%s",
                this->msMeasure.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,
                "Does the average %s differ from zero?",
                this->msMeasure.c_str());
        question = strdup(tmpstr);
      }
      else if (ncf == 1)
      {
        QdecFactor* contFactor = this->mContinuousFactors[nthcf-1];
        string contFactorName = contFactor->GetFactorName();
        sprintf(tmpstr,
                "Avg-%s-%s-Cor",
                this->msMeasure.c_str(),
                contFactorName.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,
                "Does the correlation between %s and %s, accounting for %s, "
                "differ from zero?",
                this->msMeasure.c_str(),
                contFactorName.c_str(),
                df1Name.c_str());
        question = strdup(tmpstr);
      }
      else if (ncf == 2)
      {
        QdecFactor* contFactor = this->mContinuousFactors[nthcf-1];
        string contFactorName = contFactor->GetFactorName();
        QdecFactor* otherFactor;
        if (nthcf == 1) otherFactor = this->mContinuousFactors[nthcf];
        else otherFactor = this->mContinuousFactors[nthcf-2];
        string otherContFactorName = otherFactor->GetFactorName();
        sprintf(tmpstr,
                "Avg-%s-%s-Cor",
                this->msMeasure.c_str(),
                contFactorName.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,
                "Does the correlation between %s and %s, accounting for %s "
                "and %s, differ from zero?",
                this->msMeasure.c_str(),
                contFactorName.c_str(),
                df1Name.c_str(),
                otherContFactorName.c_str());
        question = strdup(tmpstr);
      }
      QdecContrast* newContrast = new QdecContrast ( contrast,
                                                     name,
                                                     question );
      this->mContrasts.push_back( newContrast );
    }
    // Now do the Within-Factor (Simple Main) Effects
    for (unsigned int nthcf = 0; nthcf < (ncf+1); nthcf++)
    {
      vector< double > contrast;
      for (unsigned int i=0; i < nreg; i++) contrast.push_back( 0.0 );
      assert( contrast.size() == nreg );
      int a = 2*nthcf;
      contrast[a] = 1;
      contrast[a+1] = -1;
      string name = "";
      string question = "";
      if (nthcf == 0)
      {
        sprintf(tmpstr,"Diff-%s-%s-Intercept-%s",
                df1l1name,df1l2name,this->msMeasure.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,"Does the average %s differ between %s and %s?",
                this->msMeasure.c_str(),df1l1name,df1l2name);
        question = strdup(tmpstr);
      }
      else if (ncf == 1)
      {
        QdecFactor* contFactor = this->mContinuousFactors[nthcf-1];
        string contFactorName = contFactor->GetFactorName();
        sprintf(tmpstr,"Diff-%s-%s-Cor-%s-%s",
                df1l1name,df1l2name,
                this->msMeasure.c_str(),
                contFactorName.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,"Does the %s--%s correlation differ between %s and %s?",
                this->msMeasure.c_str(),
                contFactorName.c_str(),
                df1l1name,df1l2name);
        question = strdup(tmpstr);
      }
      else if (ncf == 2)
      {
        QdecFactor* contFactor = this->mContinuousFactors[nthcf-1];
        string contFactorName = contFactor->GetFactorName();
        QdecFactor* otherFactor;
        if (nthcf == 1) otherFactor = this->mContinuousFactors[nthcf];
        else otherFactor = this->mContinuousFactors[nthcf-2];
        string otherContFactorName = otherFactor->GetFactorName();
        sprintf(tmpstr,"Diff-%s-%s-Cor-%s-%s",
                df1l1name,df1l2name,
                this->msMeasure.c_str(),
                contFactorName.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,"Does the %s--%s correlation, accounting for %s, "
                "differ between %s and %s?",
                this->msMeasure.c_str(),
                contFactorName.c_str(),
                otherContFactorName.c_str(),
                df1l1name,df1l2name);
        question = strdup(tmpstr);
      }
      QdecContrast* newContrast = new QdecContrast ( contrast,
                                                     name,
                                                     question );
      this->mContrasts.push_back( newContrast );
    }
  }

  /*------------------------------------------------------------*/
  if (ndf == 2)
  {
    QdecFactor* df1 = this->mDiscreteFactors[0];
    string df1Name = df1->GetFactorName();
    int nl1 = df1->GetLevelNames().size();
    if (nl1 != 2)
    {
      fprintf(stderr, "ERROR: QdecGlmDesign::GenerateContrasts: "
              "factor 1 must have 2 levels\n");
      return -1;
    }
    const char* df1name = df1->GetFactorName().c_str();
    vector<string> df1LevelNames = df1->GetLevelNames();
    const char* df1l1name = df1LevelNames[0].c_str();
    const char* df1l2name = df1LevelNames[1].c_str();

    QdecFactor* df2 = this->mDiscreteFactors[1];
    string df2Name = df2->GetFactorName();
    int nl2 = df2->GetLevelNames().size();
    if (nl2 != 2)
    {
      fprintf(stderr, "ERROR: QdecGlmDesign::GenerateContrasts: "
              "factor 2 must have 2 levels\n");
      return -1;
    }
    const char* df2name = df2->GetFactorName().c_str();
    vector<string> df2LevelNames = df2->GetLevelNames();
    const char* df2l1name = df2LevelNames[0].c_str();
    const char* df2l2name = df2LevelNames[1].c_str();

    // Do the Main Effects first
    for (unsigned int nthcf = 0; nthcf < (ncf+1); nthcf++)
    {
      vector< double > contrast;
      for (unsigned int i=0; i < nreg; i++) contrast.push_back( 0.0 );
      assert( contrast.size() == nreg );
      int a = 4*nthcf;
      for (unsigned int d=0;d<4;d++) contrast[a+d] = 1;
      string name = "";
      string question = "";
      if (nthcf == 0)
      {
        sprintf(tmpstr,
                "Avg-Intercept-%s",
                this->msMeasure.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,"Does the average %s differ from zero?",
                this->msMeasure.c_str());
        question = strdup(tmpstr);
      }
      else if (ncf == 1)
      {
        QdecFactor* contFactor = this->mContinuousFactors[nthcf-1];
        string contFactorName = contFactor->GetFactorName();
        sprintf(tmpstr,
                "Avg-%s-%s-Cor",
                this->msMeasure.c_str(),
                contFactorName.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,
                "Does the correlation between %s and %s, accounting for %s "
                "and %s, differ from zero?",
                this->msMeasure.c_str(),
                contFactorName.c_str(),
                df1Name.c_str(),df2Name.c_str());
        question = strdup(tmpstr);
      }
      else if (ncf == 2)
      {
        QdecFactor* contFactor = this->mContinuousFactors[nthcf-1];
        string contFactorName = contFactor->GetFactorName();
        QdecFactor* otherFactor;
        if (nthcf == 1) otherFactor = this->mContinuousFactors[nthcf];
        else otherFactor = this->mContinuousFactors[nthcf-2];
        string otherContFactorName = otherFactor->GetFactorName();
        sprintf(tmpstr,
                "Avg-%s-%s-Cor",
                this->msMeasure.c_str(),
                contFactorName.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,
                "Does the correlation between %s and %s, accounting for %s "
                "and %s and %s, differ from zero?",
                this->msMeasure.c_str(),
                contFactorName.c_str(),
                df1Name.c_str(),df2Name.c_str(),
                otherContFactorName.c_str());
        question = strdup(tmpstr);
      }
      QdecContrast* newContrast = new QdecContrast ( contrast,
                                                     name,
                                                     question );
      this->mContrasts.push_back( newContrast );
    }
    // Now do the Within-Factor-1 (Simple Main) Effects
    for (unsigned int nthcf = 0; nthcf < (ncf+1); nthcf++)
    {
      vector< double > contrast;
      for (unsigned int i=0; i < nreg; i++) contrast.push_back( 0.0 );
      assert( contrast.size() == nreg );
      int a = 4*nthcf;
      contrast[a]   = +1;
      contrast[a+1] = -1;
      contrast[a+2] = +1;
      contrast[a+3] = -1;
      string name = "";
      string question = "";
      if (nthcf == 0)
      {
        sprintf(tmpstr,"Diff-%s-%s-Intercept-%s",
                df1l1name,df1l2name,this->msMeasure.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,"Does the average %s, accounting for %s, "
                "differ between %s and %s?",
                this->msMeasure.c_str(),
                df2Name.c_str(),
                df1l1name,df1l2name);
        question = strdup(tmpstr);
      }
      else if (ncf == 1)
      {
        QdecFactor* contFactor = this->mContinuousFactors[nthcf-1];
        string contFactorName = contFactor->GetFactorName();
        sprintf(tmpstr,"Diff-%s-%s-Cor-%s-%s",
                df1l1name,df1l2name,
                this->msMeasure.c_str(),
                contFactorName.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,
                "Does the %s--%s correlation, accounting for %s, "
                "differ between %s and %s?",
                this->msMeasure.c_str(),
                contFactorName.c_str(),
                df2Name.c_str(),
                df1l1name,df1l2name);
        question = strdup(tmpstr);
      }
      else if (ncf == 2)
      {
        QdecFactor* contFactor = this->mContinuousFactors[nthcf-1];
        string contFactorName = contFactor->GetFactorName();
        QdecFactor* otherFactor;
        if (nthcf == 1) otherFactor = this->mContinuousFactors[nthcf];
        else otherFactor = this->mContinuousFactors[nthcf-2];
        string otherContFactorName = otherFactor->GetFactorName();
        sprintf(tmpstr,"Diff-%s-%s-Cor-%s-%s",
                df1l1name,df1l2name,
                this->msMeasure.c_str(),
                contFactorName.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,
                "Does the %s--%s correlation, accounting for %s and %s, "
                "differ between %s and %s?",
                this->msMeasure.c_str(),
                contFactorName.c_str(),
                df2Name.c_str(),
                otherContFactorName.c_str(),
                df1l1name,df1l2name);
        question = strdup(tmpstr);
      }
      QdecContrast* newContrast = new QdecContrast ( contrast,
                                                     name,
                                                     question );
      this->mContrasts.push_back( newContrast );
    }
    // Now do the Within-Factor-2 (Simple Main) Effects
    for (unsigned int nthcf = 0; nthcf < (ncf+1); nthcf++)
    {
      vector< double > contrast;
      for (unsigned int i=0; i < nreg; i++) contrast.push_back( 0.0 );
      assert( contrast.size() == nreg );
      int a = 4*nthcf;
      contrast[a]   = +1;
      contrast[a+1] = +1;
      contrast[a+2] = -1;
      contrast[a+3] = -1;
      string name = "";
      string question = "";
      if (nthcf == 0)
      {
        sprintf(tmpstr,"Diff-%s-%s-Intercept-%s",
                df2l1name,df2l2name,this->msMeasure.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,"Does the average %s, accounting for %s, "
                "differ between %s and %s?",
                this->msMeasure.c_str(),
                df1Name.c_str(),
                df2l1name,df2l2name);
        question = strdup(tmpstr);
      }
      else if (ncf == 1)
      {
        QdecFactor* contFactor = this->mContinuousFactors[nthcf-1];
        string contFactorName = contFactor->GetFactorName();
        sprintf(tmpstr,"Diff-%s-%s-Cor-%s-%s",
                df2l1name,df2l2name,
                this->msMeasure.c_str(),
                contFactorName.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,
                "Does the %s--%s correlation, accounting for %s, "
                "differ between %s and %s?",
                this->msMeasure.c_str(),
                contFactorName.c_str(),
                df1Name.c_str(),
                df2l1name,df2l2name);
        question = strdup(tmpstr);
      }
      else if (ncf == 2)
      {
        QdecFactor* contFactor = this->mContinuousFactors[nthcf-1];
        string contFactorName = contFactor->GetFactorName();
        QdecFactor* otherFactor;
        if (nthcf == 1) otherFactor = this->mContinuousFactors[nthcf];
        else otherFactor = this->mContinuousFactors[nthcf-2];
        string otherContFactorName = otherFactor->GetFactorName();
        sprintf(tmpstr,"Diff-%s-%s-Cor-%s-%s",
                df2l1name,df2l2name,
                this->msMeasure.c_str(),
                contFactorName.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,
                "Does the %s--%s correlation, accounting for %s and %s, "
                "differ between %s and %s?",
                this->msMeasure.c_str(),
                contFactorName.c_str(),
                df1Name.c_str(),
                otherContFactorName.c_str(),
                df2l1name,df2l2name);
        question = strdup(tmpstr);
      }
      QdecContrast* newContrast = new QdecContrast ( contrast,
                                                     name,
                                                     question );
      this->mContrasts.push_back( newContrast );
    }
    // Now do the Interactions
    for (unsigned int nthcf = 0; nthcf < (ncf+1); nthcf++)
    {
      vector< double > contrast;
      for (unsigned int i=0; i < nreg; i++) contrast.push_back( 0.0 );
      assert( contrast.size() == nreg );
      int a = 4*nthcf;
      contrast[a]   = +1;
      contrast[a+1] = -1;
      contrast[a+2] = -1;
      contrast[a+3] = +1;
      string name = "";
      string question = "";
      if (nthcf == 0)
      {
        sprintf(tmpstr,"X-%s-%s-Intercept-%s",
                df1name,df2name,this->msMeasure.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr,"Is there a %s--%s interaction in the mean %s?",
                df1name,df2name,this->msMeasure.c_str());
        question = strdup(tmpstr);
      }
      else if (ncf == 1)
      {
        QdecFactor* contFactor = this->mContinuousFactors[nthcf-1];
        string contFactorName = contFactor->GetFactorName();
        sprintf(tmpstr,
                "X-%s-%s-Cor-%s-%s",
                df1name,df2name,
                this->msMeasure.c_str(),
                contFactorName.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr, 
                "Is there a %s--%s interaction in the %s--%s correlation?",
                df1name,df2name,
                this->msMeasure.c_str(),
                contFactorName.c_str());
        question = strdup(tmpstr);
      }
      else if (ncf == 2)
      {
        QdecFactor* contFactor = this->mContinuousFactors[nthcf-1];
        string contFactorName = contFactor->GetFactorName();
        QdecFactor* otherFactor;
        if (nthcf == 1) otherFactor = this->mContinuousFactors[nthcf];
        else otherFactor = this->mContinuousFactors[nthcf-2];
        string otherContFactorName = otherFactor->GetFactorName();
        sprintf(tmpstr,
                "X-%s-%s-Cor-%s-%s",
                df1name,df2name,
                this->msMeasure.c_str(),
                contFactorName.c_str());
        name = strdup(tmpstr);
        sprintf(tmpstr, 
                "Is there a %s--%s interaction, accounting for %s, "
                "in the %s--%s correlation?",
                df1name,df2name,
                otherContFactorName.c_str(),
                this->msMeasure.c_str(),
                contFactorName.c_str());
        question = strdup(tmpstr);
      }
      QdecContrast* newContrast = new QdecContrast ( contrast,
                                                     name,
                                                     question );
      this->mContrasts.push_back( newContrast );
    }
  }

  return 0;
}

int QdecGlmDesign::WriteContrastMatrices () 
{
  if( !this->IsValid() ) {
    fprintf( stderr, 
             "ERROR: QdecGlmDesign::WriteContrastMatrices: "
             "Design parameters not valid.\n" );
    return(-1);
  }
  
  // print all the contrasts we created, and write-out each .mat file
  for( unsigned int i=0; i < this->mContrasts.size(); i++)
  {
    QdecContrast* contrast = this->mContrasts[i];
    fprintf(stdout,"%s -----------------------\n",contrast->GetName().c_str());
    fprintf(stdout,"%s\n",contrast->GetQuestion().c_str());
    fprintf(stdout,"%s\n",contrast->GetContrastStr().c_str());
    fflush(stdout);
    
    if( contrast->WriteDotMatFile( this->mfnWorkingDir ) ) return -1;
  }

  return 0;
}



/**
 * Creates the 'y' input data to mri_glmfit, by concatenating the
 * subject volumes, and writes it to the specified filename (single volume).
 * @return int
 */
int QdecGlmDesign::WriteYdataFile ( )
{
  if( !this->IsValid() ) {
    fprintf
      ( stderr, 
        "ERROR: QdecGlmDesign::WriteYdataFile: "
        "Design parameters not valid.\n" );
    return(-1);
  }

  vector<string> lSubjectIDs = this->mDataTable->GetSubjectIDs();

  // Now we have a list of subject names. We want to concatenate the
  // files:
  // SUBJECTS_DIR/$subject/surf/$hemi.$measure.fwhm$smoothness.fsaverage.mgh
  // into a single y.mgh. First we build the file names and go
  // through all of them, making sure they exist. We'll do an
  // MRIinfo on them and check to see their dimensions are all the
  // same. Then we'll make a volume of that size and copy in the
  // header from the first input volume. Then we copy each file into
  // the dest at an incrementing frame.

  // Check the input files to see if they exist. Add the filenames
  // to a vector.
  if( this->mProgressUpdateGUI )
  {
    this->mProgressUpdateGUI->UpdateProgressMessage( "Verifying subjects..." );
    this->mProgressUpdateGUI->UpdateProgressPercent( 30 );
  }
  vector<string> lfnInputs;
  for ( vector<string>::iterator tSubjectID = lSubjectIDs.begin();
        tSubjectID != lSubjectIDs.end(); ++tSubjectID )
  {
    // If this name is in our list of subject exclusions, skip it.
    if( maExcludedSubjects.find( *tSubjectID ) !=
        maExcludedSubjects.end() ) {
      continue;
    }

    // Build file name.
    stringstream fnInput;
    fnInput << this->mfnSubjectsDir
            << "/" << *tSubjectID << "/surf/"
            << this->GetHemi() << "." 
            << this->GetMeasure() << ".fwhm"
            << this->GetSmoothness() << "." 
            << this->msAverageSubject
            << ".mgh";

    // Check to it exists and is readable.
    ifstream fInput( fnInput.str().c_str(), std::ios::in );
    if( !fInput || fInput.bad() )
      throw runtime_error( string("Couldn't open file " ) + fnInput.str() );

    // Add it to our list.
    lfnInputs.push_back( fnInput.str() );
  }

  if ( lfnInputs.size() < 1 )
    throw runtime_error( "No input files" );

  // Go through and concatenate copy all the volumes.
  if( this->mProgressUpdateGUI )
  {
    this->mProgressUpdateGUI->UpdateProgressMessage( "Concatenating volumes..." );
    this->mProgressUpdateGUI->UpdateProgressPercent( 50 );
  }

  stringstream ssCommand;
  ssCommand << "mri_concat ";
  // subject inputs...
  vector<string>::iterator tfnInput;
  for ( tfnInput = lfnInputs.begin();
        tfnInput != lfnInputs.end(); ++tfnInput )
  {
    ssCommand << *tfnInput << " ";
  }
  // and the output filename...
  ssCommand << "--o " << this->GetYdataFileName();

  // Now run the command.
  char* sCommand = strdup( ssCommand.str().c_str() );
  fflush(stdout);fflush(stderr);
  int rRun = system( sCommand );
  if ( -1 == rRun )
    throw runtime_error( "system call failed: " + ssCommand.str() );
  if ( rRun > 0 )
    throw runtime_error( "command failed: " + ssCommand.str() );
  free( sCommand );

  return 0;
}

vector<QdecFactor*> const&
QdecGlmDesign::GetDiscreteFactors () const {
  return mDiscreteFactors;
}


vector<QdecFactor*> const& 
QdecGlmDesign::GetContinuousFactors () const {
  return mContinuousFactors;
}
