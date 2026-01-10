This is the same file as the .docx, but you can't read that using VS Code. 

--- 
## First impression
When analysing this case, it immediately becomes clear how strongly human biases can influence technological systems. The core of the issue lies in the fact that users of the app are asked to rate each other’s photos. These ratings often contain implicit prejudices, for example about ethnicity, appearance, or visible disabilities. The Breeze algorithm then takes these ratings as input, reproducing the bias hidden within them. In this way, existing social prejudices are transferred into the digital domain. Although the algorithm appears to function objectively, it carries forward the inequality embedded in the data.

A second observation concerns the impact this process has on opportunities for different groups. Because the ratings are directly incorporated into the algorithm, some users systematically receive less visibility. As a result, they also have fewer chances to be matched, even though this has nothing to do with their actual suitability as partners. Instead of reducing inequality, the algorithm amplifies it. What begins with individual preferences and biases ultimately ends in unequal opportunities and a less inclusive user experience.

## DAG diagram
![DAG showing bias flow](<DAG diagram.png> "DAG diagram")

## Reflection DAG
After drawing the DAG, I realized there are aspects that I did not capture in my first impression. Initially, I focused mainly on the transfer of human bias into the algorithm and the unequal opportunities that resulted from this. What the DAG made more visible is the reinforcing mechanism: the outcomes of the algorithm (less visibility and fewer matches) can in turn strengthen people’s perceptions and biases. This creates a feedback effect that I did not fully consider at first.

Another aspect that became clearer is the role of the data itself. It is not only the users’ ratings that matter, but also how those ratings are transformed into training data. By highlighting this step in the DAG, it becomes easier to see where interventions could be applied, for example by adjusting the data or the algorithm to reduce discrimination.

## Advice
If a data scientist receives an assignment like this, the first step should be to look beyond accuracy and efficiency and actively consider fairness and inclusivity. It is important to check whether the data contains human or societal biases, and to test the algorithm’s outcomes for unequal treatment of different groups. A good practice is to build in fairness metrics, such as demographic parity or equal opportunity, alongside traditional performance metrics. In addition, the data scientist should involve diverse stakeholders early on, so that ethical concerns are identified before the system is deployed. This way, technical choices are aligned with social responsibility.
