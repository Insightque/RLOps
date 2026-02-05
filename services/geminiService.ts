
import { GoogleGenAI } from "@google/genai";
import { TrainingMetric } from "../types";

export const analyzeTrainingPerformance = async (metrics: TrainingMetric[]) => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const recentMetrics = metrics.slice(-20);
  
  const prompt = `
    As an RLOps Engineer, analyze the following DDPG training metrics and provide insights.
    Metrics Data: ${JSON.stringify(recentMetrics)}
    
    Tasks:
    1. Identify if the Critic and Actor losses are converging.
    2. Check if the average reward is increasing.
    3. Suggest potential hyperparameter adjustments (learning rate, tau, noise).
    4. Provide a summary of the current training health.
    
    Return the response in a structured format.
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: prompt,
      config: {
        thinkingConfig: { thinkingBudget: 2000 }
      }
    });
    return response.text;
  } catch (error) {
    console.error("Gemini analysis failed:", error);
    return "Failed to analyze training data. Please check connection.";
  }
};
