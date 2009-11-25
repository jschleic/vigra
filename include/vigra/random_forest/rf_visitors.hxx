/************************************************************************/
/*                                                                      */
/*        Copyright 2008-2009 by  Ullrich Koethe and Rahul Nair         */
/*                                                                      */
/*    This file is part of the VIGRA computer vision library.           */
/*    The VIGRA Website is                                              */
/*        http://kogs-www.informatik.uni-hamburg.de/~koethe/vigra/      */
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
#ifndef RF_VISITORS_HXX
#define RF_VISITORS_HXX

namespace vigra
{

    
    
/** Base Class from which all Visitors derive
 */
class VisitorBase
{
    public:
    bool active_;   
    bool is_active()
    {
        return active_;
    }

    bool has_value()
    {
        return false;
    }

    VisitorBase()
        : active_(true)
    {}

    void deactivate()
    {
        active_ = false;
    }
    void activate()
    {
        active_ = true;
    }
    
    /** do something after the the Split has decided how to process the Region
     * (Stack entry)
     *
     * \param tree      reference to the tree that is currently being learned
     * \param split     reference to the split object
     * \param parent    current stack entry  which was used to decide the split
     * \param leftChild left stack entry that will be pushed
     * \param rightChild
     *                  right stack entry that will be pushed.
     * \sa RF_Traits::StackEntry_t
     */
    template<class Tree, class Split, class Region, class Feature_t, class Label_t>
    void visit_after_split( Tree          & tree, 
                            Split         & split,
                            Region        & parent,
                            Region        & leftChild,
                            Region        & rightChild,
                            Feature_t     & features,
                            Label_t       & labels)
    {}
    
    /** do something after each tree has been learned
     *
     * \param rf        reference to the random forest object that called this
     *                  visitor
     * \param pr        reference to the preprocessor that processed the input
     * \param sm        reference to the sampler object
     * \param st        reference to the first stack entry
     * \param index     index of current tree
     */
    template<class RF, class PR, class SM, class ST>
    void visit_after_tree(RF& rf, PR & pr,  SM & sm, ST & st, int index)
    {}
    
    /** do something after all trees have been learned
     *
     * \param rf        reference to the random forest object that called this
     *                  visitor
     * \param pr        reference to the preprocessor that processed the input
     */
    template<class RF, class PR>
    void visit_at_end(RF const & rf, PR const & pr)
    {}
    
    /** do something before learning starts 
     *
     * \param rf        reference to the random forest object that called this
     *                  visitor
     */
    template<class RF, class PR>
    void visit_at_beginning(RF const & rf, PR const & pr)
    {}
    /** do some thing while traversing tree after it has been learned 
     *  (external nodes)
     *
     * \param tr        reference to the tree object that called this visitor
     * \param index     index in the topology_ array we currently are at
     * \param node_tag  type of node we have (will be e_.... - )
     * \sa  NodeTags;
     *
     * you can create the node by using a switch on node_tag and using the 
     * corresponding Node objects. Or - if you do not care about the type 
     * use the Nodebase class.
     */
    template<class TR, class IntT, class TopT>
    void visit_external_node(TR & tr, IntT index, TopT node_t)
    {}
    
    /** do something when visiting a internal node after it has been learned
     *
     * \sa visit_external_node
     */
    template<class TR, class IntT, class TopT>
    void visit_internal_node(TR & tr, IntT index, TopT node_t)
    {}

    /** return a double value.  The value of the first 
     * visitor encountered that has a return value is returned with the
     * RandomForest::learn() method - or -1.0 if no return value visitor
     * existed. This functionality basically only exists so that the 
     * OOB - visitor can return the oob error rate like in the old version 
     * of the random forest.
     */
    double return_val()
    {
        return -1.0;
    }
};

namespace rf
{

/** Last Visitor that should be called to stop the recursion.
 */
class StopVisiting: public VisitorBase
{
    public:
    bool has_value()
    {
        return true;
    }
    double return_val()
    {
        return -1.0;
    }
};
/** Container elements of the statically linked Visitor list.
 *
 * use the create_visitor() factory functions to create visitors up to size 10;
 *
 */
template <class Visitor, class Next = StopVisiting>
class VisitorNode
{
    public:
    
    StopVisiting    stop_;
    Next &          next_;
    Visitor &       visitor_;   
    VisitorNode(Visitor & visitor, Next & next) 
    : 
        next_(next), visitor_(visitor)
    {}

    VisitorNode(Visitor &  visitor) 
    : 
        next_(stop_), visitor_(visitor)
    {}

    template<class Tree, class Split, class Region, class Feature_t, class Label_t>
    void visit_after_split( Tree          & tree, 
                            Split         & split,
                            Region        & parent,
                            Region        & leftChild,
                            Region        & rightChild,
                            Feature_t     & features,
                            Label_t       & labels)
    {
        if(visitor_.is_active())
            visitor_.visit_after_split(tree, split, 
                                       parent, leftChild, rightChild,
                                       features, labels);
        next_.visit_after_split(tree, split, parent, leftChild, rightChild,
                                features, labels);
    }

    template<class RF, class PR, class SM, class ST>
    void visit_after_tree(RF& rf, PR & pr,  SM & sm, ST & st, int index)
    {
        if(visitor_.is_active())
            visitor_.visit_after_tree(rf, pr, sm, st, index);
        next_.visit_after_tree(rf, pr, sm, st, index);
    }

    template<class RF, class PR>
    void visit_at_beginning(RF & rf, PR & pr)
    {
        if(visitor_.is_active())
            visitor_.visit_at_beginning(rf, pr);
        next_.visit_at_beginning(rf, pr);
    }
    template<class RF, class PR>
    void visit_at_end(RF & rf, PR & pr)
    {
        if(visitor_.is_active())
            visitor_.visit_at_end(rf, pr);
        next_.visit_at_end(rf, pr);
    }
    
    template<class TR, class IntT, class TopT>
    void visit_external_node(TR & tr, IntT & index, TopT & node_t)
    {
        if(visitor_.is_active())
            visitor_.visit_external_node(tr, index, node_t);
        next_.visit_external_node(tr, index, node_t);
    }
    template<class TR, class IntT, class TopT>
    void visit_internal_node(TR & tr, IntT & index, TopT & node_t)
    {
        if(visitor_.is_active())
            visitor_.visit_internal_node(tr, index, node_t);
        next_.visit_internal_node(tr, index, node_t);
    }

    double return_val()
    {
        if(visitor_.is_active() && visitor_.has_value())
            return visitor_.return_val();
        return next_.return_val();
    }
};

} //namespace rf

//////////////////////////////////////////////////////////////////////////////
//  Visitor Factory function up to 10 visitors                              //
//////////////////////////////////////////////////////////////////////////////
template<class A>
rf::VisitorNode<A>
create_visitor(A & a)
{
   typedef rf::VisitorNode<A> _0_t;
   _0_t _0(a);
   return _0;
}


template<class A, class B>
rf::VisitorNode<A, rf::VisitorNode<B> >
create_visitor(A & a, B & b)
{
   typedef rf::VisitorNode<B> _1_t;
   _1_t _1(b);
   typedef rf::VisitorNode<A, _1_t> _0_t;
   _0_t _0(a, _1);
   return _0;
}


template<class A, class B, class C>
rf::VisitorNode<A, rf::VisitorNode<B, rf::VisitorNode<C> > >
create_visitor(A & a, B & b, C & c)
{
   typedef rf::VisitorNode<C> _2_t;
   _2_t _2(c);
   typedef rf::VisitorNode<B, _2_t> _1_t;
   _1_t _1(b, _2);
   typedef rf::VisitorNode<A, _1_t> _0_t;
   _0_t _0(a, _1);
   return _0;
}


template<class A, class B, class C, class D>
rf::VisitorNode<A, rf::VisitorNode<B, rf::VisitorNode<C, 
    rf::VisitorNode<D> > > >
create_visitor(A & a, B & b, C & c, D & d)
{
   typedef rf::VisitorNode<D> _3_t;
   _3_t _3(d);
   typedef rf::VisitorNode<C, _3_t> _2_t;
   _2_t _2(c, _3);
   typedef rf::VisitorNode<B, _2_t> _1_t;
   _1_t _1(b, _2);
   typedef rf::VisitorNode<A, _1_t> _0_t;
   _0_t _0(a, _1);
   return _0;
}


template<class A, class B, class C, class D, class E>
rf::VisitorNode<A, rf::VisitorNode<B, rf::VisitorNode<C, 
    rf::VisitorNode<D, rf::VisitorNode<E> > > > >
create_visitor(A & a, B & b, C & c, 
               D & d, E & e)
{
   typedef rf::VisitorNode<E> _4_t;
   _4_t _4(e);
   typedef rf::VisitorNode<D, _4_t> _3_t;
   _3_t _3(d, _4);
   typedef rf::VisitorNode<C, _3_t> _2_t;
   _2_t _2(c, _3);
   typedef rf::VisitorNode<B, _2_t> _1_t;
   _1_t _1(b, _2);
   typedef rf::VisitorNode<A, _1_t> _0_t;
   _0_t _0(a, _1);
   return _0;
}


template<class A, class B, class C, class D, class E,
         class F>
rf::VisitorNode<A, rf::VisitorNode<B, rf::VisitorNode<C, 
    rf::VisitorNode<D, rf::VisitorNode<E, rf::VisitorNode<F> > > > > >
create_visitor(A & a, B & b, C & c, 
               D & d, E & e, F & f)
{
   typedef rf::VisitorNode<F> _5_t;
   _5_t _5(f);
   typedef rf::VisitorNode<E, _5_t> _4_t;
   _4_t _4(e, _5);
   typedef rf::VisitorNode<D, _4_t> _3_t;
   _3_t _3(d, _4);
   typedef rf::VisitorNode<C, _3_t> _2_t;
   _2_t _2(c, _3);
   typedef rf::VisitorNode<B, _2_t> _1_t;
   _1_t _1(b, _2);
   typedef rf::VisitorNode<A, _1_t> _0_t;
   _0_t _0(a, _1);
   return _0;
}


template<class A, class B, class C, class D, class E,
         class F, class G>
rf::VisitorNode<A, rf::VisitorNode<B, rf::VisitorNode<C, 
    rf::VisitorNode<D, rf::VisitorNode<E, rf::VisitorNode<F, 
    rf::VisitorNode<G> > > > > > >
create_visitor(A & a, B & b, C & c, 
               D & d, E & e, F & f, G & g)
{
   typedef rf::VisitorNode<G> _6_t;
   _6_t _6(g);
   typedef rf::VisitorNode<F, _6_t> _5_t;
   _5_t _5(f, _6);
   typedef rf::VisitorNode<E, _5_t> _4_t;
   _4_t _4(e, _5);
   typedef rf::VisitorNode<D, _4_t> _3_t;
   _3_t _3(d, _4);
   typedef rf::VisitorNode<C, _3_t> _2_t;
   _2_t _2(c, _3);
   typedef rf::VisitorNode<B, _2_t> _1_t;
   _1_t _1(b, _2);
   typedef rf::VisitorNode<A, _1_t> _0_t;
   _0_t _0(a, _1);
   return _0;
}


template<class A, class B, class C, class D, class E,
         class F, class G, class H>
rf::VisitorNode<A, rf::VisitorNode<B, rf::VisitorNode<C, 
    rf::VisitorNode<D, rf::VisitorNode<E, rf::VisitorNode<F, 
    rf::VisitorNode<G, rf::VisitorNode<H> > > > > > > >
create_visitor(A & a, B & b, C & c, 
               D & d, E & e, F & f, 
               G & g, H & h)
{
   typedef rf::VisitorNode<H> _7_t;
   _7_t _7(h);
   typedef rf::VisitorNode<G, _7_t> _6_t;
   _6_t _6(g, _7);
   typedef rf::VisitorNode<F, _6_t> _5_t;
   _5_t _5(f, _6);
   typedef rf::VisitorNode<E, _5_t> _4_t;
   _4_t _4(e, _5);
   typedef rf::VisitorNode<D, _4_t> _3_t;
   _3_t _3(d, _4);
   typedef rf::VisitorNode<C, _3_t> _2_t;
   _2_t _2(c, _3);
   typedef rf::VisitorNode<B, _2_t> _1_t;
   _1_t _1(b, _2);
   typedef rf::VisitorNode<A, _1_t> _0_t;
   _0_t _0(a, _1);
   return _0;
}


template<class A, class B, class C, class D, class E,
         class F, class G, class H, class I>
rf::VisitorNode<A, rf::VisitorNode<B, rf::VisitorNode<C, 
    rf::VisitorNode<D, rf::VisitorNode<E, rf::VisitorNode<F, 
    rf::VisitorNode<G, rf::VisitorNode<H, rf::VisitorNode<I> > > > > > > > >
create_visitor(A & a, B & b, C & c, 
               D & d, E & e, F & f, 
               G & g, H & h, I & i)
{
   typedef rf::VisitorNode<I> _8_t;
   _8_t _8(i);
   typedef rf::VisitorNode<H, _8_t> _7_t;
   _7_t _7(h, _8);
   typedef rf::VisitorNode<G, _7_t> _6_t;
   _6_t _6(g, _7);
   typedef rf::VisitorNode<F, _6_t> _5_t;
   _5_t _5(f, _6);
   typedef rf::VisitorNode<E, _5_t> _4_t;
   _4_t _4(e, _5);
   typedef rf::VisitorNode<D, _4_t> _3_t;
   _3_t _3(d, _4);
   typedef rf::VisitorNode<C, _3_t> _2_t;
   _2_t _2(c, _3);
   typedef rf::VisitorNode<B, _2_t> _1_t;
   _1_t _1(b, _2);
   typedef rf::VisitorNode<A, _1_t> _0_t;
   _0_t _0(a, _1);
   return _0;
}

template<class A, class B, class C, class D, class E,
         class F, class G, class H, class I, class J>
rf::VisitorNode<A, rf::VisitorNode<B, rf::VisitorNode<C, 
    rf::VisitorNode<D, rf::VisitorNode<E, rf::VisitorNode<F, 
    rf::VisitorNode<G, rf::VisitorNode<H, rf::VisitorNode<I,
    rf::VisitorNode<J> > > > > > > > > >
create_visitor(A & a, B & b, C & c, 
               D & d, E & e, F & f, 
               G & g, H & h, I & i,
               J & j)
{
   typedef rf::VisitorNode<J> _9_t;
   _9_t _9(j);
   typedef rf::VisitorNode<I, _9_t> _8_t;
   _8_t _8(i, _9);
   typedef rf::VisitorNode<H, _8_t> _7_t;
   _7_t _7(h, _8);
   typedef rf::VisitorNode<G, _7_t> _6_t;
   _6_t _6(g, _7);
   typedef rf::VisitorNode<F, _6_t> _5_t;
   _5_t _5(f, _6);
   typedef rf::VisitorNode<E, _5_t> _4_t;
   _4_t _4(e, _5);
   typedef rf::VisitorNode<D, _4_t> _3_t;
   _3_t _3(d, _4);
   typedef rf::VisitorNode<C, _3_t> _2_t;
   _2_t _2(c, _3);
   typedef rf::VisitorNode<B, _2_t> _1_t;
   _1_t _1(b, _2);
   typedef rf::VisitorNode<A, _1_t> _0_t;
   _0_t _0(a, _1);
   return _0;
}

//////////////////////////////////////////////////////////////////////////////
// Visitors of communal interest. Do not spam this file with stuff          //
// nobody wants.                                                            //
//////////////////////////////////////////////////////////////////////////////





/** Visitor that calculates the oob error of the random forest. 
 * this is the default visitor used. 
 *
 * To bored to comment each line of this class - trust me it works.
 */
class OOB_Visitor:public VisitorBase
{
public:
    double oobError;
    int totalOobCount;
    ArrayVector<int> oobCount,oobErrorCount;

    OOB_Visitor()
    : oobError(0.0),
      totalOobCount(0)
    {}


    bool has_value()
    {
        return true;
    }
    /** does the basic calculation per tree*/
    template<class RF, class PR, class SM, class ST>
    void visit_after_tree(    RF& rf, PR & pr,  SM & sm, ST & st, int index)
    {
        //do the first time called.
        if(int(oobCount.size()) != rf.ext_param_.row_count_)
        {
            oobCount.resize(rf.ext_param_.row_count_, 0);
            oobErrorCount.resize(rf.ext_param_.row_count_, 0);
        }
        // go through the samples
        for(int l = 0; l < rf.ext_param_.row_count_; ++l)
        {
            // if the lth sample is oob...
            if(!sm.is_used()[l])
            {
                ++oobCount[l];
                if(     rf.tree(index)
                            .predictLabel(rowVector(pr.features(), l)) 
                    !=  pr.response()(l,0))
                {
                    ++oobErrorCount[l];
                }
            }

        }
    }

    /** Does the normalisation
     */
    template<class RF, class PR>
    void visit_at_end(RF & rf, PR & pr)
    {
        // do some normalisation
        for(int l=0; l < (int)rf.ext_param_.row_count_; ++l)
        if(oobCount[l])
        {
            oobError += double(oobErrorCount[l]) / oobCount[l];
            ++totalOobCount;
        }
    }
    
    //returns value of the learn function. 
    double return_val()
    {
        return oobError/totalOobCount;
    }
};


/** calculate variable importance while learning.
 */
class VariableImportanceVisitor : public VisitorBase
{
    public:

    /** This Array has the same entries as the R - random forest variable
     *  importance
     */
    MultiArray<2, double>       variable_importance_;
    int                         repetition_count_;
    bool                        in_place_;

    VariableImportanceVisitor(int rep_cnt = 10) 
    :   repetition_count_(rep_cnt)

    {}

    /** calculates impurity decrease based variable importance after every
     * split.  
     */
    template<class Tree, class Split, class Region, class Feature_t, class Label_t>
    void visit_after_split( Tree          & tree, 
                            Split         & split,
                            Region        & parent,
                            Region        & leftChild,
                            Region        & rightChild,
                            Feature_t     & features,
                            Label_t       & labels)
    {
        //resize to right size when called the first time
        
        Int32 const  class_count = tree.ext_param_.class_count_;
        Int32 const  column_count = tree.ext_param_.column_count_;
        if(variable_importance_.size() == 0)
        {
            
            variable_importance_
                .reshape(MultiArrayShape<2>::type(column_count, 
                                                 class_count+2));
        }

        if(split.createNode().typeID() == i_ThresholdNode)
        {
            Node<i_ThresholdNode> node(split.createNode());
            variable_importance_(node.column(),class_count+1) 
                += split.region_gini_ - split.minGini();
        }
    }

    /**compute permutation based var imp. 
     * (Only an Array of size oob_sample_count x 1 is created.
     *  - apposed to oob_sample_count x feature_count in the other method.
     * 
     * \sa FieldProxy
     */
    template<class RF, class PR, class SM, class ST>
    void after_tree_ip_impl(RF& rf, PR & pr,  SM & sm, ST & st, int index)
    {
        typedef MultiArrayShape<2>::type Shp_t;
        Int32                   column_count = rf.ext_param_.column_count_;
        Int32                   class_count  = rf.ext_param_.class_count_;  
        
        // remove the const cast on the features (yep , I know what I am 
        // doing here.) data is not destroyed.
        typename PR::Feature_t & features 
            = const_cast<typename PR::Feature_t &>(pr.features());

        //find the oob indices of current tree. 
        ArrayVector<Int32>      oob_indices;
        ArrayVector<Int32>::iterator
                                iter;
        for(int ii = 0; ii < rf.ext_param_.row_count_; ++ii)
            if(!sm.is_used()[ii])
                oob_indices.push_back(ii);

        //create space to back up a column      
        std::vector<double>     backup_column;

        // Random foo
#ifdef CLASSIFIER_TEST
        RandomMT19937           random(1);
#else 
        RandomMT19937           random(RandomSeed);
#endif
        UniformIntRandomFunctor<RandomMT19937>  
                                randint(random);


        //make some space for the results
        MultiArray<2, double>
                    oob_right(Shp_t(1, class_count + 1)); 
        MultiArray<2, double>
                    perm_oob_right (Shp_t(1, class_count + 1)); 
            
        
        // get the oob success rate with the original samples
        for(iter = oob_indices.begin(); 
            iter != oob_indices.end(); 
            ++iter)
        {
            if(rf.tree(index)
                    .predictLabel(rowVector(features, *iter)) 
                ==  pr.response()(*iter, 0))
            {
                //per class
                ++oob_right[pr.response()(*iter,0)];
                //total
                ++oob_right[class_count];
            }
        }
        //get the oob rate after permuting the ii'th dimension.
        for(int ii = 0; ii < column_count; ++ii)
        {
            perm_oob_right.init(0.0); 
            //make backup of orinal column
            backup_column.clear();
            for(iter = oob_indices.begin(); 
                iter != oob_indices.end(); 
                ++iter)
            {
                backup_column.push_back(features(*iter,ii));
            }
            
            //get the oob rate after permuting the ii'th dimension.
            for(int rr = 0; rr < repetition_count_; ++rr)
            {               
                //permute dimension. 
                int n = oob_indices.size();
                for(int jj = 1; jj < n; ++jj)
                    std::swap(features(oob_indices[jj], ii), 
                              features(oob_indices[randint(jj+1)], ii));

                //get the oob sucess rate after permuting
                for(iter = oob_indices.begin(); 
                    iter != oob_indices.end(); 
                    ++iter)
                {
                    if(rf.tree(index)
                            .predictLabel(rowVector(features, *iter)) 
                        ==  pr.response()(*iter, 0))
                    {
                        //per class
                        ++perm_oob_right[pr.response()(*iter, 0)];
                        //total
                        ++perm_oob_right[class_count];
                    }
                }
            }
            
            
            //normalise and add to the variable_importance array.
            perm_oob_right  /=  repetition_count_;
            perm_oob_right -=oob_right;
            perm_oob_right *= -1;
            perm_oob_right      /=  oob_indices.size();
            variable_importance_
                .subarray(Shp_t(ii,0), 
                          Shp_t(ii+1,class_count+1)) += perm_oob_right;
            //copy back permuted dimension
            for(int jj = 0; jj < int(oob_indices.size()); ++jj)
                features(oob_indices[jj], ii) = backup_column[jj];
        }
    }

    /** calculate permutation based impurity after every tree has been 
     * learned  default behaviour is that this happens out of place.
     * If you have very big data sets and want to avoid copying of data 
     * set the in_place_ flag to true. 
     */
    template<class RF, class PR, class SM, class ST>
    void visit_after_tree(RF& rf, PR & pr,  SM & sm, ST & st, int index)
    {
            after_tree_ip_impl(rf, pr, sm, st, index);
    }

    /** Normalise variable importance after the number of trees is known.
     */
    template<class RF, class PR>
    void visit_at_end(RF & rf, PR & pr)
    {
        variable_importance_ /= rf.trees_.size();
    }
};

} // namespace vigra
#endif // RF_VISITORS_HXX
