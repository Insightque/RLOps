
import { GoogleGenAI } from "@google/genai";
import { TrainingMetric } from "../types";

export const analyzeTrainingPerformance = async (metrics: TrainingMetric[]) => {
  const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
  const recentMetrics = metrics.slice(-30);
  
  const prompt = `
    As an RLOps Engineer, analyze the following DDPG training metrics and provide insights.
    Metrics Data: ${JSON.stringify(recentMetrics)}
    
    Tasks:
    1. Identify if Critic and Actor losses are converging.
    2. Check if the average reward is increasing.
    3. Suggest hyperparameter adjustments (learning rate, tau, noise).
    4. Summarize training health.
    
    Return the response as a clear Markdown report.
  `;

  try {
    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: [{ parts: [{ text: prompt }] }],
      config: {
        thinkingConfig: { thinkingBudget: 2000 }
      }
    });
    return response.text || "No analysis available.";
  } catch (error) {
    console.error("Gemini analysis failed:", error);
    return "Analysis service temporarily unavailable. Please check your connection.";
  }
};
