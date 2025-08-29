// Base URL for API requests - using Vite proxy
const API_BASE_URL = '/api';
console.log('Using API Base URL:', API_BASE_URL);

// Common headers for all requests
const defaultHeaders = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'Cache-Control': 'no-cache',
  'Pragma': 'no-cache',
  'Expires': '0'
};

export interface MessageMetadata {
  resultId?: string;
  sources?: Source[];
  evaluation?: EvaluationMetric[];
  [key: string]: any;
}

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  metadata?: MessageMetadata;
  isError?: boolean;
  isLoading?: boolean;
  timestamp?: number;
}

export interface Source {
  title: string;
  url: string;
  type: 'paper' | 'article' | 'dataset' | 'website' | string;
  relevanceScore?: number;
  snippet?: string;
  publishedDate?: string;
}

export interface EvaluationMetric {
  metric: string;
  score: number;
  description: string;
}

export interface ResearchResponse {
  success: boolean;
  status: 'success' | 'error' | 'processing';
  response: string;
  sources?: Source[];
  recommendations?: string[];
  evaluation?: EvaluationMetric[];
  metadata?: {
    queryTime?: number;
    modelUsed?: string;
    tokenCount?: number;
    [key: string]: any;
  };
}

export const researchApi = {
  async submitQuery(query: string, chatHistory: Message[] = [], file?: File): Promise<ResearchResponse> {
    // Input validation
    if (!query || typeof query !== 'string' || query.trim().length === 0) {
      throw new Error('Query cannot be empty');
    }

    // Limit query length and remove excessive whitespace
    const cleanQuery = query.trim().substring(0, 1000).replace(/\s+/g, ' ');
    
    // Check for duplicate/repetitive queries
    const lastUserMessage = chatHistory
      .filter(m => m.role === 'user')
      .pop()?.content;
      
    if (lastUserMessage && cleanQuery.includes(lastUserMessage) && 
        cleanQuery.length > lastUserMessage.length * 1.5) {
      throw new Error('Please avoid repeating your previous query. Try being more specific.');
    }
    
    console.log('Submitting query:', { query: cleanQuery, historyLength: chatHistory.length, hasFile: !!file });
    
    // Check if backend is reachable first
    try {
      console.log('Checking backend health...');
      await this.checkHealth();
      console.log('Backend health check passed');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error('Backend connection error:', error);
      throw new Error(`Unable to connect to the research service: ${errorMessage}. Please try again later.`);
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5 * 60 * 1000);

    try {
      // Format chat history to only include role and content
      const formattedChatHistory = chatHistory
        .filter(msg => msg.role !== 'system')
        .map(({ role, content }) => ({ role, content }));

      let apiUrl = `${API_BASE_URL}/research`;
      let response;
      const requestId = `req_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
      const startTime = Date.now();
      console.log('Request started at:', new Date().toISOString());
      
      if (file) {
        // Handle file upload with FormData
        const formData = new FormData();
        formData.append('query', cleanQuery);
        formData.append('file', file);
        
        // Add chat history as JSON string
        if (formattedChatHistory.length > 0) {
          formData.append('chat_history', JSON.stringify(formattedChatHistory));
        }
        
        console.log('Sending file upload request to:', apiUrl);
        response = await fetch(apiUrl, {
          method: 'POST',
          headers: {
            'Accept': 'application/json',
            'X-Request-ID': requestId
          },
          body: formData,
          signal: controller.signal,
          credentials: 'include',
        });
      } else {
        // Standard JSON request without file
        const requestBody = {
          query: cleanQuery,
          chat_history: formattedChatHistory,
        };

        console.debug('Request body:', JSON.stringify(requestBody, null, 2));
        console.log('Sending request to:', apiUrl);
        
        response = await fetch(apiUrl, {
          method: 'POST',
          headers: {
            ...defaultHeaders,
            'X-Request-ID': requestId
          },
          body: JSON.stringify(requestBody),
          signal: controller.signal,
          credentials: 'include',
        });
      }

      clearTimeout(timeoutId);
      const requestTime = Date.now() - startTime;
      console.log(`Request completed in ${requestTime}ms`);
      console.log('Received response status:', response.status);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Validate response structure
      if (typeof data.response !== 'string') {
        console.warn('Unexpected response format from API:', data);
      }

      console.debug('Response data:', data);
      return data;
      
    } catch (error) {
      clearTimeout(timeoutId);
      console.error('Error submitting research query:', error);
      
      let errorMessage = 'Failed to submit research query';
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          errorMessage = 'Request timed out. The research is taking longer than expected.';
        } else {
          errorMessage = error.message || errorMessage;
        }
      }
      
      // Return a structured error response
      return {
        success: false,
        status: 'error',
        response: errorMessage,
        metadata: {
          error: true,
          message: errorMessage,
          timestamp: new Date().toISOString(),
        }
      };
    }
  },

  async checkHealth(): Promise<{ status: string; version: string }> {
    console.log(`Checking health at: ${API_BASE_URL}/health`);
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      });
      
      console.log('Health check response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.error('Health check failed with status:', response.status, 'Response:', errorText);
        throw new Error(`Health check failed with status ${response.status}: ${errorText}`);
      }
      
      const data = await response.json();
      console.log('Health check response:', data);

      return data;
    } catch (error) {
      console.error('Backend health check failed:', error);
      throw new Error('Unable to connect to the research service. Please check if the backend server is running.');
    }
  },
};
