import React, {
  createContext,
  useContext,
  useState,
  ReactNode,
  useCallback,
  useEffect,
  useRef,
} from "react";
import axios from "axios";
import {
  Message,
  ResearchResponse,
  Source,
  EvaluationMetric,
} from "@/lib/api";

// Types for research status
type ResearchStatus = "pending" | "processing" | "completed" | "error" | "success";
type ResearchStep = "query_analysis" | "source_retrieval" | "document_processing" | "analysis" | "synthesis";

interface ResearchStatusResponse {
  success: boolean;
  status: ResearchStatus;
  step?: ResearchStep;
  message?: string;
  response?: ResearchResponse;
  metadata?: {
    progress?: number;
    start_time?: number;
    elapsed_time?: number;
    [key: string]: any;
  };
}

interface ResearchResult {
  id: string;
  timestamp: number;
  sources: Source[];
  evaluation: EvaluationMetric[];
  recommendations: string[];
  status: ResearchStatus;
  step: ResearchStep;
  progress: number;
  metadata: {
    queryTime?: number;
    modelUsed?: string;
    tokenCount?: number;
    requestId?: string;
    [key: string]: any;
  };
  response: string;
  success: boolean;
}

type ResearchContextType = {
  handleFileSelect: (file: File | null) => void;
  messages: Message[];
  currentResult: ResearchResult | null;
  isLoading: boolean;
  error: string | null;
  selectedFile: File | null;
  submitQuery: (query: string) => Promise<void>;
  submitFileQuery: (query: string, file: File) => Promise<void>;
  clearMessages: () => void;
  sources: Source[];
  evaluation: EvaluationMetric[];
  recommendations: string[];
  status: ResearchStatus;
  step: ResearchStep;
  progress: number;
  cancelResearch: () => void;
};

const ResearchContext = createContext<ResearchContextType | undefined>(undefined);

export const ResearchProvider: React.FC<{ children: ReactNode }> = ({ children }): JSX.Element => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentResult, setCurrentResult] = useState<ResearchResult | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sources, setSources] = useState<Source[]>([]);
  const [evaluation, setEvaluation] = useState<EvaluationMetric[]>([]);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [status, setStatus] = useState<ResearchStatus>("pending");
  const [step, setStep] = useState<ResearchStep>("query_analysis");
  const [progress, setProgress] = useState<number>(0);
  const [currentRequestId, setCurrentRequestId] = useState<string | null>(null);

  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const isMounted = useRef(true);

  // Cleanup
  useEffect(() => {
    return () => {
      isMounted.current = false;
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  const handleFileSelect = useCallback((file: File | null) => {
    setSelectedFile(file);
  }, []);

  const cancelResearch = useCallback(async () => {
    const requestId = currentRequestId;
    if (!requestId) return;
    
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    
    try {
      // Call the cancel endpoint
      await axios.post(`/api/v1/research/cancel/${requestId}`);
      
      setStatus("error");
      setError("Research was cancelled by user");
      setIsLoading(false);
      setCurrentRequestId(null);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "Research was cancelled by user",
          timestamp: Date.now(),
          metadata: { type: "error" },
        },
      ]);
    } catch (error) {
      console.error("Failed to cancel research:", error);
      // Still update the UI even if the backend call fails
      setStatus("error");
      setError("Research was cancelled by user");
      setIsLoading(false);
      setCurrentRequestId(null);
    }
  }, [currentRequestId]);

  const pollResearchStatus = useCallback(async (requestId: string) => {
    if (!isMounted.current) return;
    
    
    const statusUrl = `/api/v1/research/status/${requestId}`;
    console.log(`[${new Date().toISOString()}] Polling status at:`, statusUrl);
    
    try {
      const response = await fetch(statusUrl, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache'
        },
        credentials: 'include'
      });
      
      console.log(`[${new Date().toISOString()}] Status response:`, response.status, response.statusText);
      
      if (!response.ok) {
        let errorData;
        try {
          // Try to parse as JSON first
          errorData = await response.json();
          console.error(`[${new Date().toISOString()}] Error response:`, errorData);
          
          // Extract error details
          const errorMessage = errorData.detail || errorData.message || `HTTP error! status: ${response.status}`;
          const error = new Error(errorMessage);
          (error as any).details = errorData;
          (error as any).status = response.status;
          throw error;
        } catch (e) {
          // If JSON parsing fails, try to get text
          const errorText = await response.text().catch(() => 'No error text available');
          console.error(`[${new Date().toISOString()}] Error response:`, errorText);
          throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
        }
      }
      
      const statusData: ResearchStatusResponse = await response.json();
      
      // Update status and progress
      setStatus(statusData.status);
      if (statusData.step) setStep(statusData.step);
      if (statusData.metadata?.progress !== undefined) {
        setProgress(statusData.metadata.progress);
      }

      // Update loading message with current status
      setMessages(prev => {
        const lastMessage = prev[prev.length - 1];
        if (lastMessage?.isLoading) {
          return [
            ...prev.slice(0, -1),
            {
              ...lastMessage,
              content: statusData.message || `Research in progress (${statusData.step || 'processing'})...`,
              metadata: {
                ...lastMessage.metadata,
                status: statusData.status,
                step: statusData.step,
                progress: statusData.metadata?.progress || 0
              }
            }
          ];
        }
        return prev;
      });

      // Handle completion or error
      if (["completed", "error"].includes(statusData.status)) {
        // Clear polling
        if (pollingRef.current) {
          clearInterval(pollingRef.current);
          pollingRef.current = null;
        }
        
        setIsLoading(false);

        if (statusData.status === "completed" && statusData.response) {
          const result: ResearchResult = {
            id: requestId,
            timestamp: Date.now(),
            status: "completed",
            step: statusData.step || "synthesis",
            progress: 100,
            success: true,
            sources: Array.isArray(statusData.response.sources) ? statusData.response.sources : [],
            evaluation: Array.isArray(statusData.response.evaluation) ? statusData.response.evaluation : [],
            recommendations: Array.isArray(statusData.response.recommendations) ? statusData.response.recommendations : [],
            response: statusData.response.response || "Research completed",
            metadata: {
              ...(statusData.response.metadata || {}),
              requestId,
              queryTime: statusData.metadata?.elapsed_time,
            }
          };

          // Update state with results
          setCurrentResult(result);
          setSources(result.sources);
          setEvaluation(result.evaluation);
          setRecommendations(result.recommendations);

          // Replace loading message with final result
          setMessages(prev => {
            const filtered = prev.filter(msg => !msg.isLoading);
            return [
              ...filtered,
              {
                role: "assistant",
                content: result.response,
                timestamp: Date.now(),
                metadata: {
                  requestId,
                  sources: result.sources,
                  evaluation: result.evaluation,
                  status: "completed"
                },
              },
            ];
          });
        } else if (statusData.status === "error") {
          const errorMessage = statusData.message || "An error occurred during research";
          setError(errorMessage);
          
          // Extract error details if available
          const errorDetails = statusData.metadata?.error || null;
          
          // Update loading message with error
          setMessages(prev => {
            const lastMessage = prev[prev.length - 1];
            if (lastMessage?.isLoading) {
              return [
                ...prev.slice(0, -1),
                {
                  ...lastMessage,
                  content: `Error: ${errorMessage}`,
                  isLoading: false,
                  isError: true,
                  metadata: {
                    ...lastMessage.metadata,
                    status: "error",
                    error: errorMessage,
                    errorDetails: errorDetails
                  }
                }
              ];
            }
            return prev;
          });
          
          // Update current result with error information
          setCurrentResult({
            id: statusData.metadata?.requestId || `error_${Date.now()}`,
            response: `Error: ${errorMessage}`,
            success: false,
            status: "error",
            step: statusData.step || "query_analysis",
            progress: statusData.metadata?.progress || 0,
            sources: statusData.response?.sources || [],
            recommendations: statusData.response?.recommendations || [],
            evaluation: statusData.response?.evaluation || [],
            metadata: {
              error: errorMessage,
              errorDetails: errorDetails,
              timestamp: Date.now(),
              ...statusData.metadata
            },
            timestamp: Date.now()
          });
        }
      }
    } catch (err) {
      console.error("Error polling research status:", err);
      if (isMounted.current) {
        const errorMsg = err instanceof Error ? err.message : "Failed to get research status";
        setError(errorMsg);
        setIsLoading(false);
        
        // Update loading message with error
        setMessages(prev => {
          const lastMessage = prev[prev.length - 1];
          if (lastMessage?.isLoading) {
            return [
              ...prev.slice(0, -1),
              {
                ...lastMessage,
                content: `Error: ${errorMsg}`,
                isLoading: false,
                isError: true,
                metadata: {
                  ...lastMessage.metadata,
                  status: "error",
                  error: errorMsg
                }
              }
            ];
          }
          return prev;
        });
      }
    }
  }, []);

  const submitQuery = useCallback(async (query: string) => {
    if (!query.trim()) {
      setError("Please enter a valid query");
      return;
    }

    // Clear any existing polling
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }

    // Reset states
    setIsLoading(true);
    setError(null);
    setStatus("processing");
    setProgress(0);
    setSources([]);
    setEvaluation([]);
    setRecommendations([]);

    const trimmedQuery = query.trim().slice(0, 1000);
    const requestId = `req_${Date.now()}`;
    setCurrentRequestId(requestId);

    // Add user message
    const userMessage: Message = {
      role: "user",
      content: trimmedQuery,
      timestamp: Date.now(),
      metadata: { requestId },
    };

    // Add loading indicator
    const loadingMessage: Message = {
      role: "assistant",
      content: "Processing your request. This may take a moment...",
      timestamp: Date.now(),
      isLoading: true,
      metadata: { 
        requestId, 
        status: "processing", 
        step: "query_analysis", 
        progress: 0 
      },
    };

    setMessages((prev) => [...prev, userMessage, loadingMessage]);

    try {
      // Start the research
      let endpoint = "/api/v1/research";
      let body;
      let headers = { "Accept": "application/json" };

      if (selectedFile) {
        endpoint = "/api/v1/research/upload";
        const formData = new FormData();
        formData.append("query", trimmedQuery);
        formData.append("file", selectedFile);
        formData.append("chat_history", JSON.stringify(messages));
        body = formData;
      } else {
        headers["Content-Type"] = "application/json";
        body = JSON.stringify({ 
          query: trimmedQuery,
          chat_history: messages
        });
      }

      const response = await fetch(endpoint, {
        method: "POST",
        headers,
        body
      });

      if (!response.ok) {
        let errorData;
        try {
          errorData = await response.json();
        } catch (e) {
          // If we can't parse JSON, use the status text
          throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }
        
        // Extract detailed error information
        const errorMessage = errorData.detail || errorData.message || "Failed to start research";
        const errorDetail = typeof errorData === 'object' ? errorData : {};
        
        // Create a structured error object
        const error = new Error(errorMessage);
        (error as any).details = errorDetail;
        (error as any).status = response.status;
        throw error;
      }

      const data = await response.json();
      const taskId = data.metadata?.request_id || data.request_id;
      
      if (!taskId) {
        console.error("No task ID in response:", data);
        throw new Error("No task ID received from server. Please try again.");
      }

      // Start polling for updates
      console.log("Starting polling for task ID:", taskId);
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
      
      // Initial poll immediately
      await pollResearchStatus(taskId);
      
      // Then set up regular polling
      pollingRef.current = setInterval(() => {
        pollResearchStatus(taskId);
      }, 2000);
      
    } catch (err) {
      console.error("Error submitting query:", err);
      const errorMsg = err instanceof Error ? err.message : "An unknown error occurred";
      setError(errorMsg);
      setIsLoading(false);
      setStatus("error");
      
      // Extract detailed error information if available
      let errorDetails = null;
      if (err instanceof Error) {
        errorDetails = (err as any).details || null;
        if (!errorDetails && (err as any).status) {
          errorDetails = { status: (err as any).status };
        }
      }

      // Update the loading message with error
      setMessages(prev => {
        const lastMessage = prev[prev.length - 1];
        if (lastMessage?.isLoading) {
          return [
            ...prev.slice(0, -1),
            {
              ...lastMessage,
              content: `Error: ${errorMsg}`,
              isLoading: false,
              isError: true,
              metadata: {
                ...lastMessage.metadata,
                status: "error",
                error: errorMsg,
                errorDetails: errorDetails
              }
            }
          ];
        }
        return prev;
      });
      
      // Update current result with error information
      setCurrentResult({
        id: `error_${Date.now()}`,
        response: `Error: ${errorMsg}`,
        success: false,
        status: "error",
        step: "query_analysis",
        progress: 0,
        sources: [],
        recommendations: [],
        evaluation: [],
        metadata: {
          error: errorMsg,
          errorDetails: errorDetails,
          timestamp: Date.now()
        },
        timestamp: Date.now()
      });
    }
  }, [pollResearchStatus, selectedFile, messages]);

  const submitFileQuery = useCallback(async (query: string, file: File) => {
    if (!query.trim() || !file) {
      setError("Please enter a valid query and provide a file");
      return;
    }

    // Clear any existing polling
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }

    // Reset states
    setIsLoading(true);
    setError(null);
    setStatus("processing");
    setStep("document_processing"); // Start with document processing for file uploads
    setProgress(0);
    setSources([]);
    setEvaluation([]);
    setRecommendations([]);

    const trimmedQuery = query.trim().slice(0, 1000);
    const requestId = `req_${Date.now()}`;
    setCurrentRequestId(requestId);

    // Add user message with file information
    const userMessage: Message = {
      role: "user",
      content: `${trimmedQuery} [Uploaded file: ${file.name}]`,
      timestamp: Date.now(),
      metadata: { 
        requestId,
        fileUpload: {
          name: file.name,
          size: file.size,
          type: file.type
        }
      },
    };

    // Add loading indicator
    const loadingMessage: Message = {
      role: "assistant",
      content: "Processing document and researching. This may take a moment...",
      timestamp: Date.now(),
      isLoading: true,
      metadata: { 
        requestId, 
        status: "processing", 
        step: "document_processing", 
        progress: 0 
      },
    };

    setMessages((prev) => [...prev, userMessage, loadingMessage]);

    try {
      // Start the research with file upload
      const endpoint = "/api/v1/research/upload";
      const formData = new FormData();
      formData.append("query", trimmedQuery);
      formData.append("file", file);
      formData.append("chat_history", JSON.stringify(messages));

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Accept": "application/json" },
        body: formData
      });

      if (!response.ok) {
        let errorData;
        try {
          errorData = await response.json();
        } catch (e) {
          // If we can't parse JSON, use the status text
          throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }
        
        // Extract detailed error information
        const errorMessage = errorData.detail || errorData.message || "Failed to start research with file";
        const errorDetail = typeof errorData === 'object' ? errorData : {};
        
        // Create a structured error object
        const error = new Error(errorMessage);
        (error as any).details = errorDetail;
        (error as any).status = response.status;
        throw error;
      }

      const data = await response.json();
      const taskId = data.metadata?.request_id || data.request_id;
      
      if (!taskId) {
        console.error("No task ID in response:", data);
        throw new Error("No task ID received from server. Please try again.");
      }

      // Start polling for updates
      console.log("Starting polling for task ID:", taskId);
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
      
      // Initial poll immediately
      await pollResearchStatus(taskId);
      
      // Then set up regular polling
      pollingRef.current = setInterval(() => {
        pollResearchStatus(taskId);
      }, 2000);
      
    } catch (err) {
      console.error("Error submitting file query:", err);
      const errorMsg = err instanceof Error ? err.message : "An unknown error occurred";
      setError(errorMsg);
      setIsLoading(false);
      setStatus("error");
      
      // Extract detailed error information if available
      let errorDetails = null;
      if (err instanceof Error) {
        errorDetails = (err as any).details || null;
        if (!errorDetails && (err as any).status) {
          errorDetails = { status: (err as any).status };
        }
      }

      // Update the loading message with error
      setMessages(prev => {
        const lastMessage = prev[prev.length - 1];
        if (lastMessage?.isLoading) {
          return [
            ...prev.slice(0, -1),
            {
              ...lastMessage,
              content: `Error: ${errorMsg}`,
              isLoading: false,
              isError: true,
              metadata: {
                ...lastMessage.metadata,
                status: "error",
                error: errorMsg,
                errorDetails: errorDetails
              }
            }
          ];
        }
        return prev;
      });
      
      // Update current result with error information
      setCurrentResult({
        id: `error_${Date.now()}`,
        response: `Error: ${errorMsg}`,
        success: false,
        status: "error",
        step: "document_processing",
        progress: 0,
        sources: [],
        recommendations: [],
        evaluation: [],
        metadata: {
          error: errorMsg,
          errorDetails: errorDetails,
          timestamp: Date.now()
        },
        timestamp: Date.now()
      });
    }
  }, [pollResearchStatus, messages]);

  const clearMessages = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
    setMessages([]);
    setCurrentResult(null);
    setSources([]);
    setEvaluation([]);
    setRecommendations([]);
    setStatus("pending");
    setStep("query_analysis");
    setProgress(0);
    setCurrentRequestId(null);
    setError(null);
  }, []);

  return (
    <ResearchContext.Provider
      value={{
        handleFileSelect,
        selectedFile,
        messages,
        currentResult,
        isLoading,
        error,
        submitQuery,
        submitFileQuery,
        clearMessages,
        sources,
        evaluation,
        recommendations,
        status,
        step,
        progress,
        cancelResearch,
      }}
    >
      {children}
    </ResearchContext.Provider>
  );
};

export const useResearch = (): ResearchContextType => {
  const context = useContext(ResearchContext);
  if (!context) throw new Error("useResearch must be used within a ResearchProvider");
  return context;
};
