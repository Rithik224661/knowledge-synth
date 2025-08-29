import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  Search, 
  FileText, 
  Brain, 
  BarChart3, 
  CheckCircle, 
  Clock,
  AlertCircle,
  Loader2
} from "lucide-react";
import { useResearch } from "@/contexts/ResearchContext";
import { useState, useEffect } from "react";

interface ProcessStep {
  id: string;
  title: string;
  description: string;
  status: "pending" | "running" | "completed" | "error";
  tool: string;
}

interface ResearchProcessProps {
  steps: ProcessStep[];
  currentStep?: string;
}

export const ResearchProcess = () => {
  const { isLoading, status: researchStatus, step: currentStep } = useResearch();
  
  // Default steps that will be updated based on the actual research process
  const [steps, setSteps] = useState<ProcessStep[]>([
    {
      id: 'query_analysis',
      title: 'Query Analysis',
      description: 'Analyzing your research question',
      status: 'pending',
      tool: 'llm_analysis'
    },
    {
      id: 'source_retrieval',
      title: 'Source Retrieval',
      description: 'Searching for relevant sources',
      status: 'pending',
      tool: 'web_search'
    },
    {
      id: 'document_processing',
      title: 'Document Processing',
      description: 'Extracting and processing information',
      status: 'pending',
      tool: 'document_loader'
    },
    {
      id: 'analysis',
      title: 'Analysis',
      description: 'Analyzing the gathered information',
      status: 'pending',
      tool: 'llm_analysis'
    },
    {
      id: 'synthesis',
      title: 'Synthesis',
      description: 'Synthesizing the results',
      status: 'pending',
      tool: 'llm_synthesis'
    }
  ]);
  
  // Update steps based on current step from ResearchContext
  useEffect(() => {
    if (isLoading) {
      // Update steps based on currentStep from ResearchContext
      setSteps(prevSteps => {
        const newSteps = [...prevSteps];
        
        // Find the index of the current step
        const currentStepIndex = newSteps.findIndex(step => step.id === currentStep);
        
        // Update all steps based on their position relative to current step
        return newSteps.map((step, index) => {
          // Steps before current step are completed
          if (index < currentStepIndex) {
            return { ...step, status: 'completed' };
          }
          // Current step is running
          else if (index === currentStepIndex) {
            return { ...step, status: 'running' };
          }
          // Steps after current step remain pending
          else {
            return { ...step, status: 'pending' };
          }
        });
      });
    } else if (researchStatus === 'completed') {
      // Set all steps to completed when research is done
      setSteps(prevSteps => 
        prevSteps.map(step => ({
          ...step,
          status: 'completed'
        }))
      );
    }
  }, [isLoading, currentStep, researchStatus]);
  const getStepIcon = (tool: string) => {
    switch (tool) {
      case "web_search":
      case "arxiv_search":
        return <Search className="h-4 w-4" />;
      case "document_loader":
      case "pdf_parser":
        return <FileText className="h-4 w-4" />;
      case "llm_synthesis":
      case "analysis":
        return <Brain className="h-4 w-4" />;
      case "visualization":
      case "data_analysis":
        return <BarChart3 className="h-4 w-4" />;
      default:
        return <Search className="h-4 w-4" />;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-4 w-4 text-success" />;
      case "running":
        return <Clock className="h-4 w-4 text-warning animate-spin" />;
      case "error":
        return <AlertCircle className="h-4 w-4 text-destructive" />;
      default:
        return <Clock className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "completed":
        return <Badge variant="success">Completed</Badge>;
      case "running":
        return <Badge variant="warning">Running</Badge>;
      case "error":
        return <Badge variant="destructive">Error</Badge>;
      default:
        return <Badge variant="secondary">Pending</Badge>;
    }
  };

  const completedSteps = steps.filter(step => step.status === "completed").length;
  const inProgress = steps.some(step => step.status === 'running');
  const progress = inProgress 
    ? (completedSteps / steps.length) * 100 + 10 // Show some progress when running
    : (completedSteps / steps.length) * 100;

  return (
    <Card className="p-6 bg-gradient-card shadow-card">
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {inProgress ? (
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
            ) : (
              <div className="h-5 w-5 rounded-full bg-success flex items-center justify-center">
                <CheckCircle className="h-3.5 w-3.5 text-white" />
              </div>
            )}
            <h2 className="text-lg font-semibold text-foreground">
              {inProgress ? 'Research in Progress' : 'Research Complete'}
            </h2>
          </div>
          <div className="text-sm text-muted-foreground">
            {completedSteps}/{steps.length} steps completed
          </div>
        </div>

        <Progress value={progress} className="w-full" />

        <div className="space-y-4">
          {steps.map((step, index) => (
            <div
              key={step.id}
              className={`flex items-start gap-4 p-4 rounded-lg border transition-all ${
                step.status === 'running' 
                  ? 'border-primary bg-accent/50' 
                  : 'border-border bg-background/50'
              }`}
            >
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-muted flex items-center justify-center">
                {getStepIcon(step.tool)}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <h3 className="font-medium text-foreground">{step.title}</h3>
                  {getStatusBadge(step.status)}
                </div>
                <p className="text-sm text-muted-foreground mb-2">{step.description}</p>
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className="text-xs">
                    {step.tool}
                  </Badge>
                </div>
              </div>

              <div className="flex-shrink-0">
                {getStatusIcon(step.status)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
};