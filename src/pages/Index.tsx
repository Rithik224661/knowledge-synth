import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { QueryInput } from "@/components/research/QueryInput";
import { ResearchProcess } from "@/components/research/ResearchProcess";
import { ResultsDisplay } from "@/components/research/ResultsDisplay";
import { 
  Brain, 
  Search, 
  FileText, 
  BarChart3,
  Sparkles,
  Users,
  Target,
  Zap
} from "lucide-react";
import heroImage from "@/assets/research-hero.jpg";

const Index = () => {
  const [isResearching, setIsResearching] = useState(false);
  const [showResults, setShowResults] = useState(false);

  // Mock research process steps
  const mockSteps = [
    {
      id: "1",
      title: "Query Decomposition",
      description: "Breaking down complex query into actionable sub-tasks",
      status: "completed" as const,
      tool: "llm_analysis"
    },
    {
      id: "2", 
      title: "ArXiv Paper Search",
      description: "Searching for latest LLM safety research papers",
      status: "completed" as const,
      tool: "arxiv_search"
    },
    {
      id: "3",
      title: "Document Analysis", 
      description: "Extracting and analyzing content from research papers",
      status: "running" as const,
      tool: "document_loader"
    },
    {
      id: "4",
      title: "Dataset Identification",
      description: "Finding and evaluating mentioned datasets",
      status: "pending" as const,
      tool: "web_search"
    },
    {
      id: "5",
      title: "Trend Analysis",
      description: "Analyzing publication trends over the past year",
      status: "pending" as const,
      tool: "data_analysis"
    },
    {
      id: "6",
      title: "Final Synthesis",
      description: "Generating comprehensive research summary",
      status: "pending" as const,
      tool: "llm_synthesis"
    }
  ];

  // Mock research results
  const mockResults = {
    summary: "Based on analysis of recent LLM safety research, three key papers have emerged: 'Constitutional AI: Harmlessness from AI Feedback' (Anthropic, 2023), 'Red Teaming Language Models to Reduce Harms' (DeepMind, 2023), and 'Measuring and Narrowing the Compositionality Gap in Language Models' (OpenAI, 2023). These papers focus on alignment techniques, adversarial testing, and compositional reasoning safety. The HarmBench dataset from the first paper is recommended for its comprehensive coverage of potential harm categories.",
    sources: [
      {
        title: "Constitutional AI: Harmlessness from AI Feedback",
        url: "https://arxiv.org/abs/2212.08073",
        type: "paper" as const,
        relevanceScore: 0.95
      },
      {
        title: "Red Teaming Language Models to Reduce Harms",
        url: "https://arxiv.org/abs/2209.07858", 
        type: "paper" as const,
        relevanceScore: 0.88
      },
      {
        title: "HarmBench: A Standardized Evaluation Framework",
        url: "https://github.com/center-for-ai-safety/HarmBench",
        type: "dataset" as const,
        relevanceScore: 0.92
      }
    ],
    recommendations: [
      "Use HarmBench dataset for comprehensive safety evaluation - contains 510 adversarial prompts across 18 harm categories",
      "Implement Constitutional AI techniques for self-supervision in safety alignment",
      "Consider red-teaming approaches early in model development lifecycle"
    ],
    evaluation: [
      {
        metric: "Answer Faithfulness",
        score: 0.89,
        description: "Generated answer is well-grounded in retrieved sources with minimal hallucination"
      },
      {
        metric: "Answer Relevance", 
        score: 0.94,
        description: "Response directly addresses all parts of the original query comprehensively"
      },
      {
        metric: "Tool Use Accuracy",
        score: 0.87,
        description: "Agent selected appropriate tools and parameters for each sub-task"
      }
    ]
  };

  const handleQuerySubmit = (query: string) => {
    setIsResearching(true);
    // Simulate research process
    setTimeout(() => {
      setShowResults(true);
      setIsResearching(false);
    }, 3000);
  };

  const features = [
    {
      icon: <Brain className="h-6 w-6" />,
      title: "Multi-Tool Orchestration",
      description: "Intelligently coordinates web search, document analysis, and LLM synthesis"
    },
    {
      icon: <Target className="h-6 w-6" />,
      title: "Query Decomposition", 
      description: "Breaks complex requests into actionable sub-tasks using advanced reasoning"
    },
    {
      icon: <BarChart3 className="h-6 w-6" />,
      title: "Performance Evaluation",
      description: "Built-in faithfulness, relevance, and tool accuracy assessment"
    },
    {
      icon: <Zap className="h-6 w-6" />,
      title: "Real-time Processing",
      description: "Live updates on research progress with transparent tool execution"
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-gradient-hero">
        <div className="absolute inset-0">
          <img 
            src={heroImage} 
            alt="AI Research Assistant Platform" 
            className="w-full h-full object-cover opacity-20"
          />
        </div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-24">
          <div className="text-center space-y-8">
            <div className="space-y-4">
              <Badge variant="outline" className="mb-4">
                <Sparkles className="h-3 w-3 mr-1" />
                AI-Powered Research Assistant
              </Badge>
              <h1 className="text-4xl md:text-6xl font-bold text-foreground">
                Multi-Tool AI Research
                <span className="bg-gradient-research bg-clip-text text-transparent"> Assistant</span>
              </h1>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
                Advanced agent framework that orchestrates web search, document analysis, and LLM synthesis 
                to deliver comprehensive, evaluated research insights for founders and scientists.
              </p>
            </div>
            
            <div className="flex flex-wrap justify-center gap-4">
              <Button variant="research" size="lg">
                <Brain className="h-5 w-5 mr-2" />
                Start Research
              </Button>
              <Button variant="outline" size="lg">
                <FileText className="h-5 w-5 mr-2" />
                View Documentation
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-16 bg-muted/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-foreground mb-4">Core Capabilities</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Built on LangChain with sophisticated orchestration, evaluation, and multi-tool integration
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <Card key={index} className="p-6 text-center bg-gradient-card shadow-card hover:shadow-elevated transition-shadow">
                <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mx-auto mb-4 text-primary">
                  {feature.icon}
                </div>
                <h3 className="font-semibold text-foreground mb-2">{feature.title}</h3>
                <p className="text-sm text-muted-foreground">{feature.description}</p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Main Interface */}
      <section className="py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-8">
            <div className="space-y-6">
              <QueryInput onSubmit={handleQuerySubmit} isLoading={isResearching} />
              
              {(isResearching || showResults) && (
                <ResearchProcess 
                  steps={mockSteps} 
                  currentStep={isResearching ? "3" : undefined}
                />
              )}
            </div>

            <div className="space-y-6">
              {showResults ? (
                <ResultsDisplay result={mockResults} />
              ) : (
                <Card className="p-8 bg-gradient-card shadow-card text-center">
                  <Search className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                  <h3 className="font-semibold text-foreground mb-2">Ready for Research</h3>
                  <p className="text-muted-foreground">
                    Submit a complex research query to see the multi-tool agent in action.
                    Results will appear here with full evaluation metrics.
                  </p>
                </Card>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Backend Notice */}
      <section className="py-12 bg-accent/20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="p-6 bg-card rounded-lg border border-border shadow-card">
            <Users className="h-8 w-8 text-primary mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-foreground mb-2">Backend Integration Required</h3>
            <p className="text-muted-foreground mb-4">
              To enable full functionality including LangChain orchestration, LLM calls, web search, 
              and document processing, connect this project to Supabase using our native integration.
            </p>
            <p className="text-sm text-muted-foreground">
              Click the green Supabase button in the top right to get started with backend functionality.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Index;