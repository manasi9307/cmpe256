package com.predictionmarketing1.recommemderitem;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.eval.IRStatistics;
import org.apache.mahout.cf.taste.eval.RecommenderBuilder;
import org.apache.mahout.cf.taste.eval.RecommenderEvaluator;
import org.apache.mahout.cf.taste.eval.RecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;
import org.apache.mahout.cf.taste.impl.eval.GenericRecommenderIRStatsEvaluator;
import org.apache.mahout.cf.taste.impl.eval.RMSRecommenderEvaluator;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.recommender.GenericItemBasedRecommender;
//import org.apache.mahout.cf.taste.impl.similarity.EuclideanDistanceSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.LogLikelihoodSimilarity;
import org.apache.mahout.cf.taste.impl.similarity.TanimotoCoefficientSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.ItemSimilarity;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) throws TasteException, Exception
    {
    	PrintWriter fout = new PrintWriter(new BufferedWriter(new FileWriter("C:\\Users\\manas\\Desktop\\Data\\prediction11.dat")));
    	DataModel model = new FileDataModel (new File("C:\\Users\\manas\\Desktop\\Data\\train.csv"));
    	final ItemSimilarity itemSimilarity = new TanimotoCoefficientSimilarity(model);
    	
    	Recommender itemRecommender = new GenericItemBasedRecommender(model,itemSimilarity);
    	RecommenderBuilder recommenderBuilder = new RecommenderBuilder() {
			public Recommender buildRecommender(DataModel model)  {                

				return new GenericItemBasedRecommender(model,itemSimilarity);                
			}
		};
		long uId=0;
		long iId=0;
		DataModel model2 = new FileDataModel(new File("C:\\Users\\manas\\Desktop\\Data\\test.csv"));
		for( LongPrimitiveIterator users=model2.getUserIDs(); users.hasNext();) {
			uId=users.nextLong();
			for(LongPrimitiveIterator items=model2.getItemIDsFromUser(uId).iterator();
					items.hasNext();) {
				iId=items.nextLong();
				float preference = itemRecommender.estimatePreference(uId,iId);
				if(Float.isNaN(preference)) {
					fout.println(3);

				}else {

					fout.println((int)Math.round(preference));

				}
}
}
		fout.close();
		RecommenderEvaluator evaluator = new RMSRecommenderEvaluator();        
		double score = evaluator.evaluate(recommenderBuilder, null, model, 0.7, 1.0);    
		System.out.println("RMSE: " + score);

		RecommenderIRStatsEvaluator statsEvaluator = new GenericRecommenderIRStatsEvaluator();        
		IRStatistics stats = statsEvaluator.evaluate(recommenderBuilder, null, model, null, 10, 4, 0.7); // evaluate precision recall at 10

		System.out.println("Precision: " + stats.getPrecision());
		System.out.println("Recall: " + stats.getRecall());
		System.out.println("F1 Score: " + stats.getF1Measure()); 
}
}
