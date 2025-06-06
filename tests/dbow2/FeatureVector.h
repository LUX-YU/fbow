/**
 * File: Featurevector.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: feature std::vector
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_FEATURE_vector__
#define __D_T_FEATURE_vector__

#include "BowVector.h"
#include <map>
#include <vector>
#include <iostream>

namespace DBoW2 {

/// std::vector of nodes with indexes of local features
class Featurevector: 
  public std::map<NodeId, std::vector<unsigned int> >
{
public:

  /**
   * Constructor
   */
  Featurevector(void);
  
  /**
   * Destructor
   */
  ~Featurevector(void);
  
  /**
   * Adds a feature to an existing node, or adds a new node with an initial
   * feature
   * @param id node id to add or to modify
   * @param i_feature index of feature to add to the given node
   */
  void addFeature(NodeId id, unsigned int i_feature){
	    Featurevector::iterator vit = this->lower_bound(id);
  
  if(vit != this->end() && vit->first == id)
  {
    vit->second.push_back(i_feature);
  }
  else
  {
    vit = this->insert(vit, Featurevector::value_type(id, 
      std::vector<unsigned int>() ));
    vit->second.push_back(i_feature);
  }
}

  /**
   * Sends a string versions of the feature std::vector through the stream
   * @param out stream
   * @param v feature std::vector
   */
  friend std::ostream& operator<<(std::ostream &out, const Featurevector &v){
	   if(!v.empty())
  {
    Featurevector::const_iterator vit = v.begin();
    
    const std::vector<unsigned int>* f = &vit->second;

    out << "<" << vit->first << ": [";
    if(!f->empty()) out << (*f)[0];
    for(unsigned int i = 1; i < f->size(); ++i)
    {
      out << ", " << (*f)[i];
    }
    out << "]>";
    
    for(++vit; vit != v.end(); ++vit)
    {
      f = &vit->second;
      
      out << ", <" << vit->first << ": [";
      if(!f->empty()) out << (*f)[0];
      for(unsigned int i = 1; i < f->size(); ++i)
      {
        out << ", " << (*f)[i];
      }
      out << "]>";
    }
  }
  
  return out;  
}
    
};

} // namespace DBoW2

#endif

